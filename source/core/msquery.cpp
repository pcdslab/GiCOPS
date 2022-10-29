/*
 * Copyright (C) 2022 Muhammad Haseeb, Fahad Saeed
 * Florida International University, Miami, FL
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more detailSpectrum.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */
#include <dirent.h>
#include "msquery.hpp"
#include "cuda/superstep2/kernel.hpp"

using namespace std;

#define BIN_BATCHSIZE               50000
#define TEMPVECTOR_SIZE             KBYTES(20)

extern gParams params;

#ifdef USE_MPI
// MPI datatype for info_t
MPI_Datatype MPI_info;
// handle for summary.dbprep file
MPI_File fh;

#else
// handle for summary.dbprep file
std::ofstream *fh;

#endif // USE_MPI


MSQuery::MSQuery()
{
    qfile = nullptr;
    currPtr = 0;
    curr_chunk = 0;
    running_count = 0;
    qfileIndex = 0;
    m_isinit = false;
}

MSQuery::~MSQuery()
{
    currPtr = 0;
    qfileIndex = 0;
    curr_chunk = 0;
    running_count = 0;
    m_isinit = false;

    if (qfile != NULL)
    {
        delete qfile;
        qfile = NULL;
    }

    if (params.filetype == gParams::FileType_t::MS2)
        spectrum.deallocate();

    spectrum.SpectrumSize = 0;
    spectrum.prec_mz = 0;
    spectrum.Z = 0;
    spectrum.rtime = 0;
}

//
// function to read MS2 files
//
std::array<int, 2> MSQuery::readMS2file(string *filename)
{
    /* Get a new ifstream object and open file */
    ifstream *qqfile = new ifstream(*filename);

    int_t largestspec = 0;
    int_t count = 0;
    int_t specsize = 0;
    /* Check if file opened */
    if (qqfile->is_open() /*&& status =SLM_SUCCESS*/)
    {
        string_t line;

        /* While we still have lines in MS2 file */
        while (!qqfile->eof())
        {
            /* Read one line */
            getline(*qqfile, line);

            /* Empty line */
            if (line.empty())
            {
                continue;
            }
            /* Scan: (S) */
            else if (line[0] == 'S')
            {
                count++;
                largestspec = max(specsize, largestspec);
                specsize = 0;
            }
            /* Header: (H) */
            else if (line[0] == 'H' || line[0] == 'I' || line[0] == 'D' || line[0] == 'Z')
            {
                /* TODO: Decide what to do with header */
                continue;
            }
            /* MS/MS data: [m/z] [int] */
            else
            {
                specsize++;
            }
        }

        // check if the last spectrum in the file is the largest
        largestspec = max(specsize, largestspec);

        /* Close the file */
        qqfile->close();

        delete qqfile;

    }
    else
        cout << "Error: Unable to open qqfile: " << *filename << endl;

    largestspec = max(specsize, largestspec);

    return std::array<int, 2>{count, largestspec};
}

std::array<int, 2> MSQuery::convertAndprepMS2bin(string *filename)
{
    int_t largestspec = 0;
    int_t count = 0;
    int_t globalcount = 0;
    int_t specsize = 0;

    char_t *Zsave;
    char_t *Isave;

    std::vector<spectype_t> mzs;
    std::vector<spectype_t> intns;

    // reverse 20 * 1024 vector length
    mzs.reserve(TEMPVECTOR_SIZE);
    intns.reserve(TEMPVECTOR_SIZE);

    spectype_t *m_mzs = new int[BIN_BATCHSIZE * QALEN];
    spectype_t *m_intns = new spectype_t[BIN_BATCHSIZE * QALEN];

    int    m_idx = 0;

    float *rtimes = new float[2 * BIN_BATCHSIZE];
    float *prec_mz = rtimes + BIN_BATCHSIZE;

    int *z = new int[2 * BIN_BATCHSIZE];
    int *lens = z + BIN_BATCHSIZE;

    /* Get a new ifstream object and open file */
    ifstream *qqfile = new ifstream(*filename);

    /* Check if file opened */
    if (qqfile->is_open())
    {
        string_t line;
        bool isFirst = true;

        /* While we still have lines in MS2 file */
        while (!qqfile->eof())
        {
            /* Read one line */
            getline(*qqfile, line);

            if (line.empty() || line[0] == 'H' || line[0] == 'D')
            {
                continue;
            }
            /* Scan: (S) */
            else if (line[0] == 'S')
            {
                if (!isFirst)
                {
                    // largest spectrum size
                    largestspec = max(specsize, largestspec);

                    // specsize will update here
                    MSQuery::pickpeaks(mzs, intns, specsize, m_idx, m_intns, m_mzs);

                    // write the updated specsize
                    lens[count] = specsize;
                    m_idx += specsize;

                    count++;
                    globalcount++;

                    // if the buffer is full, then dump to file
                    if (count == BIN_BATCHSIZE)
                    {
                        // flush to the binary file
                        MSQuery::flushBinaryFile(filename, m_mzs, m_intns, rtimes, prec_mz, z, lens, count);

                        count = 0;
                        m_idx = 0;
                    }
                }
                else
                    isFirst = false;

                specsize = 0;

            }
            else if (line[0] == 'Z')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Zsave);
                mh = strtok_r(NULL, " \t", &Zsave);
                string_t val = "1";

                if (mh != NULL)
                    val = string_t(mh);

                z[count] = MAX(1, std::atoi(val.c_str()));

                val = "0.01";
                mh = strtok_r(NULL, " \t", &Zsave);

                if (mh != NULL)
                    val = string_t(mh);

                prec_mz[count] = std::atof(val.c_str());
            }
            else if (line[0] == 'I')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Isave);
                mh = strtok_r(NULL, " \t", &Isave);
                string_t val = "";

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                if (val.compare("RTime") == 0)
                {
                    val = "0.00";
                    mh = strtok_r(NULL, " \t", &Isave);

                    if (mh != NULL)
                    {
                        val = string_t(mh);
                    }

                    rtimes[count] = MAX(0.0, std::atof(val.c_str()));
                }
            }
            /* MS/MS data: [m/z] [int] */
            else
            {
                /* Split line into two DOUBLEs
                 * using space as delimiter */

                char_t *mz1 = strtok_r((char_t *) line.c_str(), " ", &Zsave);
                char_t *intn1 = strtok_r(NULL, " ", &Zsave);
                string_t mz = "0.01";
                string_t intn = "0.01";

                if (mz1 != NULL)
                {
                    mz = string_t(mz1);
                }

                if (intn1 != NULL)
                {
                    intn = string_t(intn1);
                }

                // integrize the values if spectype_t is int
                if constexpr (std::is_same<int, spectype_t>::value)
                {
                    mzs.push_back(std::move(std::atof(mz.c_str()) * params.scale));
                    intns.push_back(std::move(std::atof(intn.c_str()) * YAXISMULTIPLIER));
                }
                else
                {
                    mzs.push_back(std::move(std::atof(mz.c_str())));
                    intns.push_back(std::move(std::atof(intn.c_str())));
                }

                // increment the spectrum size
                specsize++;
            }
        }

        //
        // process the last spectrum in the file here
        //
        largestspec = max(specsize, largestspec);

        // specsize will update here
        MSQuery::pickpeaks(mzs, intns, specsize, m_idx, m_intns, m_mzs);

        // update lens and counts
        lens[count] = specsize;
        m_idx += specsize;

        count++;
        globalcount++;

        // flush the last batch to the file
        MSQuery::flushBinaryFile(filename, m_mzs, m_intns, rtimes, prec_mz, z, lens, count, true);

        /* Close the file */
        qqfile->close();

        delete qqfile;
    }
    else
        cout << "Error: Unable to open qqfile: " << *filename << endl;

    largestspec = max(specsize, largestspec);

    // delete the temp arrays
    delete[] m_mzs;
    delete[] rtimes;
    delete[] z;

    // return global count and largest spectrum length
    return std::array<int, 2>{globalcount, largestspec};

}

template<typename T>
status_t MSQuery::pickpeaks(std::vector<T> &mzs, std::vector<T> &intns, int &specsize, int m_idx, T *m_intns, T *m_mzs)
{
    int_t SpectrumSize = specsize;

    auto intnArr = m_intns + m_idx;
    auto mzArr = m_mzs + m_idx;

    if (SpectrumSize > 0)
    {
        KeyVal_Parallel<T, T>(intns.data(), mzs.data(), (uint_t)SpectrumSize, 1);

        // intensity normalization applied
        double factor = ((double_t) params.base_int / intns[SpectrumSize - 1]);

        // filter out intensities > params.min_int (or 1% of base peak)
        auto l_min_int = params.min_int; //0.01 * dIntArr[SpectrumSize - 1];

        // TODO: choose either one and discard the other code for intensity normalization

#if 1 // NON-STD code

        /* Set the highest peak to base intensity */
        intns[SpectrumSize - 1] = params.base_int;
        int newspeclen = 1;

        /* Scale the rest of the peaks to the base peak */
        for (int_t j = SpectrumSize - 2; j >= (SpectrumSize - QALEN) && j >= 0; j--)
        {
            intns[j] *= factor;

            if (intns[j] >= l_min_int)
                newspeclen++;
            // since sorted in ascending order. TODO: Check if this is correct
            else 
                break;
        }
#else
        // STL-based code for intensity normalization

        // beginning position of noramlization region
        auto bpos = intns.begin();

        if (SpectrumSize > QALEN)
            bpos = intns.end() - QALEN;

        // apply normalization
        std::for_each(bpos, intns.end(), [&](T &x) { x *= factor; });

        // find the position of the peak > min_int
        auto p_beg = std::lower_bound(bpos, intns.end(), l_min_int);
        int newspeclen = std::distance(p_beg, intns.end());

#endif // STL code

        /* Check the size of spectrum */
        if (newspeclen >= QALEN)
        {
            /* Copy the last QALEN elements to expSpecs */
            std::copy(mzs.end() - QALEN, mzs.end(), mzArr);
            std::copy(intns.end() - QALEN, intns.end(), intnArr);

            // if larger than QALEN then only write the last QALEN elements and set the new length
            newspeclen = QALEN;
        }
        else
        {
            /* Copy the last QALEN elements to expSpecs */
            std::copy(mzs.end() - newspeclen, mzs.end(), mzArr);
            std::copy(intns.end() - newspeclen, intns.end(), intnArr);
        }

            // assign the new length to specsize
            specsize = newspeclen;
    }
    else
    {
        std::cerr << "Spectrum size is zero" << endl;
        //std::fill(intns.begin(), intns.end(), 0);
    }

    mzs.clear();
    intns.clear();

    return SLM_SUCCESS;
}

void MSQuery::flushBinaryFile(string *filename, spectype_t *m_mzs, spectype_t *m_intns, float *rtimes, float *prec_mz, int *z, int *lens, int count, bool close)
{
    static thread_local bool_t isNewFile = true;
    static thread_local std::ofstream qbFile;

    if (isNewFile)
    {
        qbFile.open(*filename + ".pbin", ios::binary);
        isNewFile = false;
    }

    if (qbFile.is_open())
    {
        int ind = 0;

        for (int i = 0; i < count; i++)
        {
            qbFile.write((char *)&prec_mz[i], sizeof(float));
            qbFile.write((char *)&z[i], sizeof(int));
            qbFile.write((char *)&rtimes[i], sizeof(float));
            qbFile.write((char *)&lens[i], sizeof(int));

            qbFile.write((char *)&m_mzs[ind], sizeof(spectype_t) * lens[i]);
            qbFile.write((char *)&m_intns[ind], sizeof(spectype_t) * lens[i]);

            ind += lens[i];
        }
    }
    else
        std::cerr << "Could not open file " << filename << ".pbin" << std::endl;

    if (close)
    {
        qbFile.flush();
        qbFile.close();
        isNewFile = true;
    }
}

/*
 * FUNCTION:
 *
 * DESCRIPTION: Initialize structures using the query file
 *
 * INPUT:
 * @filename : Path to query file
 *
 * OUTPUT:
 * @status: Status of execution
 */
status_t MSQuery::initialize(string_t *filename, int_t fno)
{
    status_t status = SLM_SUCCESS;

    // FIXME condition at which it will depend
    auto vals = (params.filetype == gParams::FileType_t::PBIN)? convertAndprepMS2bin(filename): readMS2file(filename);

    auto largestspec = vals[1];
    auto globalcount = vals[0];

    /* Check the largestspecsize */
    if (largestspec < 1)
        status = ERR_INVLD_SIZE;
    else
    {
        currPtr  = 0;
        info.QAcount = globalcount;
        MS2file = *filename;
        curr_chunk = 0;
        running_count = 0;
        info.nqchunks = std::ceil(((double) info.QAcount / QCHUNK));
        qfileIndex = fno;
        info.maxslen = largestspec;

        // FIXME: Fix with proper conditions
        if (params.filetype == gParams::FileType_t::MS2)
            // allocate memory for the largest spectrum in file
            spectrum.allocate(info.maxslen + 1);
        else
            // binary file
            MS2file += ".pbin";

        m_isinit = true;
    }

    /* Return the status */
    return status;
}

void MSQuery::setFilename(string_t &filename) { this->MS2file = filename; }

//
// info initialized at remote process, initialize rest here
//
void MSQuery::vinitialize(string_t *filename, int_t fno)
{
    // reset these variables
    currPtr  = 0;
    curr_chunk = 0;
    running_count = 0;

    // set the file information
    MS2file = *filename;
    qfileIndex = fno;

    if (params.filetype == gParams::FileType_t::MS2)
        // allocate memory for the largest spectrum in file
        spectrum.allocate(info.maxslen + 1);
    else
        MS2file += ".pbin";

    m_isinit = true;
}

//
// initialize the binary MS2 file index if needed
//
bool MSQuery::init_index(const std::vector<string_t> &queryfiles)
{
    bool summaryExists = false;

    string_t fname = params.datapath + "/summary.dbprep";

    // function to check if a file exists
    auto fileexists = [](const string_t &fname) -> bool
    {
    // COMPILER VERSION GCC 9.1.0+ required 
#if __GNUC__ > 9 || (__GNUC__ == 9 && (__GNUC_MINOR__ >= 1))
        // COMPILER VERSION GCC 9.1.0+ required for std::filesystem calls
        auto ret = std::filesystem::exists(fname);
#else
        ifstream file(fname);
        auto ret = file.good();
        file.close();
#endif // GNUC VERSION

        return ret;
    };

    // function to verify PBIN files
    auto verifypbinfiles = [&]()
    {
        // check for all .pbin files
        auto dir = opendir(params.datapath.c_str());
        dirent* pdir;
        std::vector<string_t> pbinfiles;

        // check if opened
        if (dir != nullptr)
        {
            while ((pdir = readdir(dir)) != nullptr)
            {
                string_t cfile(pdir->d_name);
                cfile = cfile.substr(cfile.find_last_of("."));
                // Add the matching files
                if (cfile.find(".pbin") != std::string::npos)
                    pbinfiles.push_back(params.datapath + '/' + pdir->d_name);
            }
        }
        else
            return false;

        if (pbinfiles.size() == queryfiles.size())
        {
            // check if all files exist
            for (auto &ms2file : queryfiles)
            {
                string_t tempfile = ms2file + ".pbin";
                if (std::find(pbinfiles.begin(), pbinfiles.end(), tempfile) == pbinfiles.end())
                    return false;
            }
        }
        else
            return false;

        return true;
    };

    // check if file exists
    bool exists = !params.reindex && fileexists(fname) && verifypbinfiles();


    // if the file does not exist, create it
    if (!exists)
    {
        // set reindex = true
        params.reindex = true;

#ifdef USE_MPI

        // create a MPI data type
        MPI_Type_contiguous((int_t)(sizeof(info_t) / sizeof(int_t)),
                            MPI_INT,
                            &MPI_info);

        MPI_Type_commit(&MPI_info);

        // make sure all processes have the same file
        hcp::mpi::barrier();

        // open the file as ofstream file
        status_t err = MPI_File_open(MPI_COMM_WORLD, fname.c_str(), (MPI_MODE_CREATE | MPI_MODE_WRONLY), MPI_INFO_NULL, &fh);

#else

        fh = new ofstream;
        fh->open(fname, ios::out | ios::binary);

        // if unable to open fh
        if (!fh)
        {
            std::cerr << "Error opening file: " << fname << std::endl;
            exit (-1);
        }

#endif // USE_MPI
    }

    return exists;
}

status_t MSQuery::write_index()
{
#ifdef USE_MPI
    return MPI_File_close(&fh);
#else

    if (fh && fh->is_open())
    {
        fh->close();

        delete fh;
        fh = nullptr;
    }

    return SLM_SUCCESS;

#endif // USE_MPI
}

status_t MSQuery::read_index(info_t *findex, int_t count)
{
    // file name
    string fname = params.datapath + "/summary.dbprep";

#ifdef USE_MPI
    MPI_File fh2;

    // open the file
    status_t status = MPI_File_open(MPI_COMM_WORLD, fname.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh2);

    // read the index
    status = MPI_File_read_all(fh2, findex, count, MPI_info, MPI_STATUS_IGNORE);

    // close the file
    MPI_File_close(&fh2);
#else

    std::ifstream fh2(fname, ios::in | ios::binary);

    status_t status = SLM_SUCCESS;

    if (fh2.is_open())
    {
        fh2.read((char_t *)findex, count * sizeof(info_t));
        fh2.close();
    }
    else
        status = ERR_FILE_NOT_FOUND;

#endif // USE_MPI

    return status;
}

status_t MSQuery::archive(int_t index)
{
#ifdef USE_MPI
    return MPI_File_write_at(fh, sizeof(info_t)*(index), &info, 1, MPI_info, MPI_STATUS_IGNORE); 
#else

    // fh advances with every write/read
    fh->write((char_t *)&info, sizeof(info_t));

    return SLM_SUCCESS;
#endif // USE_MPI
}

template <typename T>
status_t MSQuery::extractbatch(uint_t count, Queries<T> *expSpecs, int_t &rem)
{
    status_t status = SLM_SUCCESS;

    /* half open interval [startspec, endspec) */
    uint_t startspec = running_count;
    uint_t endspec = running_count + count;

    /*if (startspec >= QAcount)
    {
        status = ERR_INVLD_SIZE;
    }

    if (status == SLM_SUCCESS) */
    {
        if (endspec > info.QAcount)
        {
            endspec = info.QAcount;
            count = endspec - startspec;
        }
    }

    expSpecs->numSpecs = count;
    expSpecs->idx[0] = 0; //Set starting point to zero.

    if (qfile == NULL || qfile->is_open() == false)
    {
        /* Get a new ifstream object and open file */
        qfile = new ifstream;

        // Open file as bin or simple text
        (params.filetype == gParams::FileType_t::PBIN)? qfile->open(MS2file, ios::in | ios::binary) : qfile->open(MS2file, ios::in);
    }

    /* Check if file opened */
    if (qfile->is_open() /*&& status =SLM_SUCCESS*/)
    {
        if (params.filetype == gParams::FileType_t::PBIN)
            readBINbatch<T>(startspec, endspec, expSpecs);
        else
        {
            for (uint_t spec = startspec; spec < endspec; spec++)
            {
                readMS2spectrum();
                status = pickpeaks(expSpecs);
            }
        }
    }
    else
    {
        std::cerr << "Error opening file: " << MS2file << std::endl;
        status = ERR_FILE_NOT_FOUND;
        exit(ERR_FILE_NOT_FOUND);
    }

    //if (status == SLM_SUCCESS)
    //{
        /* Update the runnning count */
        running_count += count;

        /* Set the number of remaining spectra count */
        rem = info.QAcount - running_count;
    //}

    return status;
}

template <typename T>
void MSQuery::readBINbatch(int startspec, int endspec, Queries<T> *expSpecs)
{
    auto prec_mz = expSpecs->precurse;
    auto z = expSpecs->charges;
    auto rtimes = expSpecs->rtimes;
    auto lens = expSpecs->idx;
    auto m_mzs = expSpecs->moz;
    auto m_intns = expSpecs->intensity;

    int ind = 0;

    if (qfile->is_open())
    {
        // temporary spectrum length variable
        int_t clen = 0;

        // lens[0] must be 0
        lens[0] = 0;

        auto count = (endspec - startspec);

        // FIXME: are loop limits correct?
        for (int i = 0; i < count; i++)
        {
            qfile->read((char *)&prec_mz[i], sizeof(float));
            qfile->read((char *)&z[i], sizeof(int));
            qfile->read((char *)&rtimes[i], sizeof(float));
            qfile->read((char *)&clen, sizeof(int));

            qfile->read((char *)&m_mzs[ind], sizeof(T) * clen);
            qfile->read((char *)&m_intns[ind], sizeof(T) * clen);

            ind += clen;
            lens[i+1] = lens[i] + clen;
            
        }
    }

    // set the total number of peaks
    expSpecs->numPeaks = ind;
}

VOID MSQuery::readMS2spectrum()
{
    string_t line;
    uint_t speclen = 0;
    char_t *saveptr;
    char_t *Isave;

    /* Check if this is the first spectrum in file */
    if (currPtr == 0)
    {
        BOOL scan = false;

        while (!qfile->eof())
        {
            /* Read one line */
            getline(*qfile, line);

            /* Empty line */
            if (line.empty() || line[0] == 'H' || line[0] == 'D')
            {
                continue;
            }
            else if (line[0] == 'Z')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &saveptr);
                mh = strtok_r(NULL, " \t", &saveptr);
                string_t val = "1";

                if (mh != NULL)
                    val = string_t(mh);

                spectrum.Z = std::atoi(val.c_str());

                val = "0.01";
                mh = strtok_r(NULL, " \t", &saveptr);

                if (mh != NULL)
                    val = string_t(mh);

                spectrum.prec_mz = (double_t)std::atof(val.c_str());
            }
            else if (line[0] == 'I')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Isave);
                mh = strtok_r(NULL, " \t", &Isave);
                string_t val = "";

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                if (val.compare("RTime") == 0)
                {
                    val = "0.00";
                    mh = strtok_r(NULL, " \t", &Isave);

                    if (mh != NULL)
                    {
                        val = string_t(mh);
                    }

                    spectrum.rtime = (double_t)std::atof(val.c_str());
                }
            }
            else if (line[0] == 'S')
            {
                if (scan == true)
                {
                    spectrum.SpectrumSize = speclen;
                    break;
                }
                else
                    scan = true;
            }
            /* Values */
            else
            {
                /* Split line into two DOUBLEs
                 * using space as delimiter */

                char_t *mz1 = strtok_r((char_t *) line.c_str(), " ", &saveptr);
                char_t *intn1 = strtok_r(NULL, " ", &saveptr);
                string_t mz = "0.01";
                string_t intn = "0.01";

                if (mz1 != NULL)
                {
                    mz = string_t(mz1);
                }

                if (intn1 != NULL)
                {
                    intn = string_t(intn1);
                }

                spectrum.mz[speclen] = (uint_t)((double_t)std::atof(mz.c_str()) * params.scale);
                spectrum.intn[speclen] = (uint_t)((double_t)std::atof(intn.c_str()) * YAXISMULTIPLIER);

                speclen++;
            }
        }

        spectrum.SpectrumSize = speclen;
    }
    /* Not the first spectrum in file */
    else
    {
        while (!qfile->eof())
        {
            /* Read one line */
            getline(*qfile, line);

            /* Empty line */
            if (line.empty() || line[0] == 'H' || line[0] == 'D')
            {
                continue;
            }
            else if (line[0] == 'Z')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &saveptr);
                mh = strtok_r(NULL, " \t", &saveptr);
                string_t val = "1";

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                spectrum.Z = std::atoi(val.c_str());

                val = "0.01";
                mh = strtok_r(NULL, " \t", &saveptr);

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                spectrum.prec_mz = (double_t)std::atof(val.c_str());

            }
            else if (line[0] == 'I')
            {
                char_t *mh = strtok_r((char_t *) line.c_str(), " \t", &Isave);
                mh = strtok_r(NULL, " \t", &Isave);
                string_t val = "";

                if (mh != NULL)
                {
                    val = string_t(mh);
                }

                if (val.compare("RTime") == 0)
                {
                    val = "0.00";
                    mh = strtok_r(NULL, " \t", &Isave);

                    if (mh != NULL)
                    {
                        val = string_t(mh);
                    }

                    spectrum.rtime = (double_t)std::atof(val.c_str());
                }
            }
            else if (line[0] == 'S')
            {
                spectrum.SpectrumSize = speclen;
                break;
            }
            /* Values */
            else
            {
                /* Split line into two DOUBLEs
                 * using space as delimiter */
                char_t *mz1 = strtok_r((char_t *) line.c_str(), " ", &saveptr);
                char_t *intn1 = strtok_r(NULL, " ", &saveptr);

                string_t mz = "0.0";
                string_t intn = "0.0";

                if (mz1 != NULL)
                {
                    mz = string_t(mz1);
                }
                if (intn1 != NULL)
                {
                    intn = string_t(intn1);
                }

                spectrum.mz[speclen] = (uint_t)((double_t)std::atof(mz.c_str()) * params.scale);
                spectrum.intn[speclen] = (uint_t)((double_t)std::atof(intn.c_str()) * YAXISMULTIPLIER);

                speclen++;
            }
        }

        spectrum.SpectrumSize = speclen;
    }
}

template <typename T>
status_t MSQuery::pickpeaks(Queries<T> *expSpecs)
{
    T *dIntArr = spectrum.intn;
    T *mzArray = spectrum.mz;
    int_t SpectrumSize = spectrum.SpectrumSize;

    // subtract running_count to get local index of expSpecs
    expSpecs->precurse[currPtr - running_count] = spectrum.prec_mz;
    expSpecs->charges[currPtr - running_count] = MAX(1, spectrum.Z);
    expSpecs->rtimes[currPtr - running_count] = MAX(0.0, spectrum.rtime);

    KeyVal_Parallel<T, T>(dIntArr, mzArray, (uint_t)SpectrumSize, 1);

    uint_t speclen = 0;
    double_t factor = 0;

    if (SpectrumSize > 0)
    {
        factor = ((double_t) params.base_int / dIntArr[SpectrumSize - 1]);

        /* Set the highest peak to base intensity */
        dIntArr[SpectrumSize - 1] = params.base_int;
        speclen = 1;

        uint_t l_min_int = params.min_int; //0.01 * dIntArr[SpectrumSize - 1];

        /* Scale the rest of the peaks to the base peak */
        for (int_t j = SpectrumSize - 2; j >= (SpectrumSize - QALEN) && j >= 0; j--)
        {
            dIntArr[j] *= factor;

            if (dIntArr[j] >= l_min_int)
            {
                speclen++;
            }
        }
    }

    /* Update the indices */
    uint_t offset = expSpecs->idx[currPtr - running_count];

    // subtract running_count to get local index of expSpecs
    expSpecs->idx[currPtr - running_count + 1] = expSpecs->idx[currPtr - running_count] + speclen;

    /* Check the size of spectrum */
    if (speclen >= QALEN)
    {
        /* Copy the last QALEN elements to expSpecs */
        std::memcpy(&expSpecs->moz[offset], (mzArray + SpectrumSize - QALEN), (QALEN * sizeof(T)));
        std::memcpy(&expSpecs->intensity[offset], (dIntArr + SpectrumSize - QALEN), (QALEN * sizeof(T)));
    }
    else
    {
        /* Copy the last speclen items to expSpecs */
        std::memcpy(&expSpecs->moz[offset], (mzArray + SpectrumSize - speclen), (speclen * sizeof(T)));
        std::memcpy(&expSpecs->intensity[offset], (dIntArr + SpectrumSize - speclen), (speclen * sizeof(T)));
    }

    expSpecs->numPeaks += speclen;

    // increase the number of spectra read
    currPtr += 1;

    return SLM_SUCCESS;
}

status_t MSQuery::DeinitQueryFile()
{
    currPtr = 0;
    info.QAcount = 0;
    info.nqchunks = 0;
    curr_chunk = 0;
    running_count = 0;
    qfileIndex = 0;
    info.maxslen = 0;

    if (qfile != NULL)
    {
        qfile->close();

        delete qfile;
        qfile = NULL;
    }

    if (params.filetype == gParams::FileType_t::MS2)
        spectrum.deallocate();

    spectrum.SpectrumSize = 0;
    spectrum.prec_mz = 0;
    spectrum.Z = 0;
    spectrum.rtime = 0;

    return SLM_SUCCESS;
}

BOOL MSQuery::isDeInit() { return ((qfile == NULL) && (info.QAcount == 0)); }

/* Operator Overload - To copy to and from the work queue */
MSQuery& MSQuery::operator=(const MSQuery &rhs)
{
    this->MS2file = rhs.MS2file;
    this->info.QAcount = rhs.info.QAcount;
    this->currPtr = rhs.currPtr;
    this->curr_chunk = rhs.curr_chunk;
    this->info.maxslen = rhs.info.maxslen;
    this->info.nqchunks = rhs.info.nqchunks;
    this->qfile = rhs.qfile;
    this->running_count = rhs.running_count;
    this->spectrum = rhs.spectrum;
    this->qfileIndex = rhs.qfileIndex;

    return *this;
}

MSQuery& MSQuery::operator=(const int_t &rhs)
{
    this->info.QAcount = rhs;
    this->currPtr = rhs;
    this->curr_chunk = rhs;
    this->info.maxslen = rhs;
    this->info.nqchunks = rhs;
    this->running_count = rhs;
    this->qfileIndex = rhs;

    return *this;
}

uint_t MSQuery::getQfileIndex() { return qfileIndex; }

uint_t MSQuery::getQAcount() { return info.QAcount; }

uint_t& MSQuery::Nqchunks() { return info.nqchunks; }

uint_t& MSQuery::Curr_chunk() { return curr_chunk; }

info_t& MSQuery::Info() { return info; }

bool_t MSQuery::isinit() { return m_isinit; }

// -------------------------------------------------------------------------------------------- //

// explicitly instantiate extractbatch with spectype_t to ensure correct instantiation
template status_t MSQuery::extractbatch<spectype_t>(uint_t, Queries<spectype_t> *, int_t &);

template status_t MSQuery::pickpeaks<spectype_t>(std::vector<spectype_t> &mzs, std::vector<spectype_t> &intns, int &specsize, int m_idx, spectype_t *m_intns, spectype_t *m_mzs);
