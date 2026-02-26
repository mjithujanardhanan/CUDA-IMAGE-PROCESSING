#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py=pybind11;


#define block_dim 16

/* this program is an execution of Black Level coreection on digital bayer domain images*/
__global__ void BLC_kernel(int* Image , int Even_0, int Even_1, int Odd_0, int Odd_1, int width, int length)
{

    int j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    int idx= (i)* width  + (j);
    if(i<length && j<width)
    {

       int offset[2][2]={{Even_0,Even_1},{Odd_0,Odd_1}};


       int val = Image[idx] - offset[i%2][j%2];

       Image[idx] = (val>0)?val:0;

    }



}





py::array_t<int> BLC(py::array_t<int> Image, py::array_t<int> BLC, int Width, int Length)   // BGGR - 0,  GBRG -1, GRBG -2, RGGB -3 
{
    auto buffer = Image.request();
    int *Input = static_cast<int*>(buffer.ptr);
    int array_size = static_cast<int>(buffer.size);
    
    auto buffer_1 = BLC.request();
    int *Black_Level = static_cast<int*>(buffer_1.ptr);

    int Even_1,Even_0,Odd_0,Odd_1;
    int *D_Image;
    cudaMalloc( &D_Image, array_size * sizeof(int));

    cudaMemcpy(D_Image , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);
    const int blockx= (Width%16 == 0)?(Width/16):(Width/16 +1),blocky= (Length%16 == 0)?(Length/16):(Length/16 +1);


    BLC_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image, Black_Level[0], Black_Level[1], Black_Level[2], Black_Level[3], Width, Length);
    cudaDeviceSynchronize();

    cudaMemcpy(Input,D_Image,array_size * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(D_Image);

    return Image;



}

PYBIND11_MODULE(BLC_module, m)
{
    m.def("BLC", &BLC, "perform Black Level Correction on an image");
}