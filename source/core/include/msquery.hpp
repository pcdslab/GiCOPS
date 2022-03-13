/*
 * Copyright (C) 2019  Muhammad Haseeb, Fahad Saeed
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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <string>
#include "common.hpp"
#include "utils.h"


// Check for integer type at compile time
template<typename T>
struct TypeIsInt
{
    static const bool value = false;
};

template<>
struct TypeIsInt<int>
{
    static const bool value = true;
};


/* Spectrum */
template <typename T>
struct _Spectrum
{
    T *mz;
    T *intn;
    uint_t SpectrumSize;
    float_t prec_mz;
    ushort_t Z;
    float_t rtime;

    // default constructor
    _Spectrum() = default;

    /* Overload the = operator - Required by MSQuery */
    _Spectrum &operator=(const _Spectrum &rhs)
    {
        this->mz = rhs.mz;
        this->intn = rhs.intn;
        this->SpectrumSize = rhs.SpectrumSize;
        this->prec_mz = rhs.prec_mz;
        this->Z = rhs.Z;
        this->rtime = rhs.rtime;

        return *this;
    }

    void allocate(uint_t size)
    {
        this->mz = new T[size];
        this->intn = new T[size];
    }

    void deallocate()
    {
        if (this->mz)
        {
            delete[] this->mz;
            this->mz = nullptr;
        }
        
        if (this->intn)
        {
            delete[] this->intn;
            this->intn = nullptr;
        }


    }
};

struct _info
{
    uint_t maxslen;       // to be archived
    uint_t nqchunks;      // to be archived
    uint_t QAcount;       // to be archived

    _info() = default;
    ~_info() = default;

    _info(uint_t _maxslen, uint_t _nqchunks, uint_t _QAcount) : maxslen(_maxslen), nqchunks(_nqchunks), QAcount(_QAcount)
    {}

    /* Overload the = operator */
    _info &operator=(const _info &rhs)
    {
        this->maxslen = rhs.maxslen;
        this->nqchunks = rhs.nqchunks;
        this->QAcount = rhs.QAcount;

        return *this;
    }

};

using info_t = _info;
using spectrum_t = _Spectrum<spectype_t>;

class MSQuery
{
protected:
    /* Global Variables */
    uint_t currPtr;
    uint_t running_count;
    uint_t curr_chunk;
    info_t info;
    std::ifstream *qfile;
    uint_t qfileIndex;
    string_t MS2file;
    spectrum_t spectrum;
    bool_t m_isinit;

    static std::array<int, 2> convertAndprepMS2bin(string_t *filename);
    static std::array<int, 2> readMS2file(string_t *filename);

    void readMS2spectrum();
    
    template <typename T>
    void readBINbatch(int, int, Queries<T> *);

    template <typename T>
    status_t pickpeaks(Queries<T> *);
    
    template <typename T>
    static status_t pickpeaks(std::vector<T> &, std::vector<T> &, int &, int, T *, T *);

public:

    MSQuery();
    ~MSQuery();
    uint_t getQAcount();
    status_t initialize(string_t *, int_t);
    void vinitialize(string_t *, int_t);
    static bool init_index();
    static status_t write_index();
    static status_t read_index(info_t *, int);
    status_t archive(int_t);

    template <typename T>
    status_t extractbatch(uint_t, Queries<T> *, int_t &);

    void setFilename(string_t &);
    status_t DeinitQueryFile();
    BOOL isDeInit();
    uint_t getQfileIndex();
    MSQuery &operator=(const MSQuery &);
    MSQuery &operator=(const int_t &);

    uint_t& Curr_chunk();
    uint_t& Nqchunks();
    info_t& Info();

    bool_t isinit();

    static void flushBinaryFile(string_t *filename, spectype_t *m_mzs, spectype_t *m_intns, float *rtimes, float *prec_mz, int *z, int *lens, int count, bool close = false);

};
