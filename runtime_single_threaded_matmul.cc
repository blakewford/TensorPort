/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>
#include <unistd.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tensorflow/compiler/xla/service/cpu/runtime_single_threaded_matmul.h"

using tensorflow::int32;
using tensorflow::int64;

namespace {

template <typename T>
void MatMul(const void* run_options_ptr, T* out, T* lhs, T* rhs, int64 m,
            int64 n, int64 k, int32 transpose_lhs, int32 transpose_rhs) {
  int64 lhs_rows = m;
  int64 lhs_cols = k;

  int64 rhs_rows = k;
  int64 rhs_cols = n;

  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Eigen::Aligned> A(
      lhs, lhs_rows, lhs_cols);
  const Eigen::TensorMap<Eigen::Tensor<const T, 2>, Eigen::Aligned> B(
      rhs, rhs_rows, rhs_cols);
  Eigen::TensorMap<Eigen::Tensor<T, 2>, Eigen::Aligned> C(out, m, n);

  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  const Eigen::array<DimPair, 1> dims(
      {DimPair(1, 0)});

  // Matrix multiply is a special case of the "contract" operation where
  // the contraction is performed along dimension 1 of the lhs and dimension
  // 0 of the rhs.
  C = A.contract(B, dims);
}

}  // namespace

int main(int argc, char** argv)
{
//  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
//  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')

    float A[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float B[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float C[sizeof(A)/sizeof(float)];

    MatMul(NULL, C, A, B, 2, 2, 3, false, false);

    char buffer[16];
    int32_t count = 2*2;
    while(count--)
    {
        memset(buffer, '\0', 16);
        sprintf(buffer, "%f\n", C[count]);
        write(1, buffer, strlen(buffer));
    }

    return 0;
}
