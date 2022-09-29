#pragma once 

#include "common.hpp"

// array to store PTM masses
__constant__ float_t modMass[ALPHABETS];

// amino acid masses
__constant__ float_t aaMass[ALPHABETS];

// static mod masses
__constant__ float_t statMass[ALPHABETS];