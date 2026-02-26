#include<cuda_runtime.h>
#include<stdio.h>
#include<cmath>
#include<stdlib.h>
#include<random>
#include<conio.h>

#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py=pybind11;


#define block_dim 16

/* this program is an execution of Defective Pixel Consealment on digital bayer domain images*/
__global__ void DP_kernel(int* Image , int* image_out, int width, int Length, int threshold)
{

    // directional gradient calculation
    long j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    long idx= (i)* width  + (j);
    if(i<Length && j<width)
    {

        int up = (i-2)<0?i+2:i-2;
        int down = (i+2)>=Length?i-2:i+2;
        int left = (j-2)<0?j+2:j-2;
        int right=(j+2)>=width?j-2:j+2;

        int p1 = up * width + j;
        int p2 = up * width + left;
        int p3 = i * width + left;
        int p4 = down * width + left;
        int p5 = down * width + j;
        int p6 = down * width + right;
        int p7 = i * width + right;
        int p8 = up * width + right;


        int d1,d2,d3,d4;
        d1 = abs(Image[p5]-Image[p1]);
        d2 = abs(Image[p6]-Image[p2]);
        d3 = abs(Image[p7]-Image[p3]);
        d4 = abs(Image[p8]-Image[p4]);

        int min=d1, neighbor_avg = (Image[p5]+Image[p1])>>1;
        if(d2<min)
        {
            min=d2;
            neighbor_avg = (Image[p6]+Image[p2])>>1;
        }
        if(d3<min)
        {
            min=d3;
            neighbor_avg = (Image[p7]+Image[p3])>>1;
        }
        if(d4<min)
        {
            min=d4;
            neighbor_avg = (Image[p8]+Image[p4])>>1;
        }

        if(abs(Image[idx] -neighbor_avg) >threshold)
        {
            image_out[idx] = neighbor_avg;
        }

        else 
        {
            image_out[idx] = Image[idx];
        }

    }



}





py::array_t<int> DPC(py::array_t<int> Image, int Width, int Length, int threshold)
{
    auto buffer = Image.request();
    int *Input = static_cast<int*>(buffer.ptr);
    long array_size = static_cast<int>(buffer.size);


    int *D_Image, *D_image_out;
    cudaMalloc( &D_Image, array_size * sizeof(int));
    cudaMalloc( &D_image_out, array_size * sizeof(int));

    cudaMemcpy(D_Image , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);
    const int blockx= (Width%16 == 0)?(Width/16):(Width/16 +1),blocky= (Length%16 == 0)?(Length/16):(Length/16 +1);


    DP_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image,D_image_out, Width, Length, threshold);
    cudaDeviceSynchronize();

    cudaMemcpy(Input,D_image_out,array_size * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(D_Image);
    cudaFree(D_image_out);

    return Image;



}

PYBIND11_MODULE(DPC_module, m)
{
    m.def("DPC", &DPC, "perform directional DPC on an image");
}