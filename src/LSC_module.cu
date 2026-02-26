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
__global__ void LSC_kernel(int* Image , int width, int Length, float gain_00, float gain_01, float gain_10, float gain_11, float Max_radius)
{

    // Lens Shading Correction calculation
    long j=blockIdx.x * blockDim.x + threadIdx.x, i=blockIdx.y * blockDim.y + threadIdx.y;
    if(i<Length && j<width)
    {
        long idx= (i)* width  + (j);
        float a[2][2]={{gain_00,gain_01},{gain_10,gain_11}};
        float dx = float(width)/2.0f -j;
        float dy = float(Length)/2.0f -i;
        float r = sqrtf(dx*dx + dy*dy);
        
        Image[idx] = (int)(Image[idx]*( 1.0f + r*a[i%2][j%2]/ Max_radius));

    }



}





py::array_t<int> LSC(py::array_t<int> Image, int Width, int Length, float gain_00, float gain_01, float gain_10, float gain_11 )  //k is the multiplication factor for gain
{
    auto buffer = Image.request();
    if(buffer.ndim != 1)
        throw std::runtime_error("image must be Flattened :: LSC module");
    if(buffer.size < (Width * Length))
        throw std::runtime_error("Wrong image size :: LSC module");
    int *Input = static_cast<int*>(buffer.ptr);
    int array_size = static_cast<int>(buffer.size);
    float max_radius = hypotf((Width)/2.0f,float(Length)/2.0f);

    int *D_Image;
    cudaMalloc( &D_Image, array_size * sizeof(int));

    cudaMemcpy(D_Image , Input, array_size * sizeof(int), cudaMemcpyHostToDevice);
    const int blockx= (Width%16 == 0)?(Width/16):(Width/16 +1),blocky= (Length%16 == 0)?(Length/16):(Length/16 +1);


    LSC_kernel<<<dim3(blockx,blocky),dim3(16,16)>>>(D_Image, Width, Length, gain_00,gain_01,gain_10,gain_11, max_radius);

    cudaMemcpy(Input,D_Image,array_size * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(D_Image);

    return Image;



}

PYBIND11_MODULE(LSC_module, m)
{
    m.def("LSC", &LSC, "perform Lens Shading Correction on the image");
}