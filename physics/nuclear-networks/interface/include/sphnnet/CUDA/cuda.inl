/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief CUDA inline preprocessor definition (used to avoid code duplication).
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */

#define COMMA ,

#if defined(__CUDACC__) || defined(__HIPCC__)
/*! @brief True if compiled with CUDA/HIP compiler */
#define COMPILE_DEVICE true

/*! @brief Host/device function decorator */
#define HOST_DEVICE_FUN __host__ __device__

/*! @brief Define variable on CPU and GPU
 *
 * @param type        variable type
 * @param symbol      variable name
 * @param definition  variable assignment (must support device assignment)
 */
#define DEVICE_DEFINE(type, symbol, definition) type symbol definition __device__ type dev_##symbol definition
#else
/*! @brief True if compiled with CUDA/HIP compiler */
#define COMPILE_DEVICE false

/*! @brief Host/device function decorator */
#define HOST_DEVICE_FUN

/*! @brief Define variable on CPU and GPU
 *
 * @param type        variable type
 * @param symbol      variable name
 * @param definition  variable assignment (must support device assignment)
 */
#define DEVICE_DEFINE(type, symbol, definition) type symbol definition
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
/*! @brief True if compiling device code */
#define DEVICE_CODE true

/*! @brief Access device-defined variable
 *
 * @param symbol   variable name to access
 */
#define DEVICE_ACCESS(symbol) dev_##symbol
#else
/*! @brief True if compiling device code */
#define DEVICE_CODE false

/*! @brief Access device-defined variable
 *
 * @param symbol   variable name to access
 */
#define DEVICE_ACCESS(symbol) symbol
#endif