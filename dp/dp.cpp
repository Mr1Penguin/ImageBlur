// dp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iomanip>

#define MAGICK 1
#define NOCACHE 2
#define IMAGE2DT 3
#define IMAGE2DT2 4
#define SUBGROUP 5
#define TEST -1

#define STATE IMAGE2DT2

#define ENABLE_WRITE 1

#if ENABLE_WRITE == 1
#define WRITEIMAGE
#endif

#define PROFILING 0

#if PROFILING == 1
#define FLAG CL_QUEUE_PROFILING_ENABLE
#else
#define FLAG 0
#endif

#define RGB 0
#define RGBA 1

#define IMAGE_FORMAT RGBA

#pragma region Filter

template<class F, class ...N>
cl::size_t<sizeof...(N) + 1> GetSizeT(F first, N... args) {
	std::vector<F> tmp{ first, static_cast<F>(args)... };
	int i = 0;
	cl::size_t<sizeof...(N) + 1> ret;
	for (auto && x : tmp)
		ret[i++] = x;
	return ret;
}

#define MagickEpsilon  (1.0e-15)
#define MagickSQ2PI  2.50662827463100024161235523934010416269302368164062
#define QuantumRange  65535.0f
#define QuantumScale  ((double) 1.0/(double) QuantumRange)

size_t GetOptimalWidth(float radius, float sigma) {
	if (radius > MagickEpsilon) {
		return size_t(2.0*ceil(radius) + 1);
	}

	auto gamma = std::abs(sigma);
	if (gamma <= MagickEpsilon) {
		return size_t{ 3 };
	}

	auto perceptible_reciprocal = [](double arg) { return std::abs(arg) >= MagickEpsilon ? 1.0 / arg : std::copysign(1.0, arg) / MagickEpsilon;  };
	auto alpha = perceptible_reciprocal(2 * gamma*gamma);
	auto beta = perceptible_reciprocal(MagickSQ2PI*gamma);
	size_t width = 5;

	while (true) {
		double normalize = 0.0;
		int j = (width - 1) / 2;
		for (int i = -j; i <= j; ++i) {
			normalize += std::exp(-double(i*i)*alpha)*beta;
		}
		auto val = std::exp(-double(j*j)*alpha)*beta / normalize;
		if (val < QuantumScale || val < MagickEpsilon) {
			break;
		}
		width += 2;
	}

	return size_t{ width - 2 };
}

std::vector<float> GenerateKernel(float radius, float sigma) {
	auto sigma_ = std::abs(sigma);
	size_t width;
	if (radius >= 1.0f) {
		width = radius * 2 + 1;
	}
	else {
		width = GetOptimalWidth(radius, sigma);
	}

	std::vector<float> kernel;
	kernel.resize(width);
	auto x = (width - 1) / 2;
	auto y = 0;

	int length = (width * 3/*rank*/ - 1) / 2;
	if (sigma > MagickEpsilon) {
		sigma *= 3;
		auto alpha = 1.0 / (2.0*sigma*sigma);
		auto beta = 1.0 / (MagickSQ2PI*sigma);
		for (int u = -length; u <= length; u++) {
			kernel[(u + length) / 3] += std::exp(-double(u*u)*alpha)*beta;
		}
	}
	else {
		kernel[x + y * width] = 1.0;
	}

	//scaling
	auto min = 0.0, max = 0.0;
	auto pos_range = 0.0, neg_range = 0.0;
	for (auto i = 0; i < width; ++i) {
		if (std::abs(kernel[i]) < MagickEpsilon) {
			kernel[i] = 0.0;
		}
		kernel[i] < 0 ? (neg_range += kernel[i]) : (pos_range += kernel[i]);
		min = *std::min_element(std::begin(kernel), std::end(kernel));
		max = *std::max_element(std::begin(kernel), std::end(kernel));
	}

	auto pos_scale = std::abs(pos_range) >= MagickEpsilon ? pos_range : 1.0;
	auto neg_scale = std::abs(neg_range) >= MagickEpsilon ? -neg_range : 1.0;

	pos_scale = 1.0 / pos_scale;
	neg_scale = 1.0 / neg_scale;

	for (int i = 0; i < width; ++i) {
		if (!std::isnan(kernel[i])) {
			kernel[i] *= kernel[i] >= 0 ? pos_scale : neg_scale;
		}
	}

	return kernel;
}

#pragma endregion

int main(int argc, char *argv[])
{
	if (argc < 6) return 1;

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	int i{ 1 };

	for (auto & pl : all_platforms) {
		std::cout << i++ << ": " << pl.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::vector<cl::Device> all_devices;
		pl.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		int j{ 1 };
		for (auto & d : all_devices) {
			std::cout << "  " << j++ << ": " << d.getInfo<CL_DEVICE_NAME>() << "; " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
		}

	}

	std::string str;
	int x, y;

	std::cout << "select device(x.y): ";
	std::cin >> str;
	x = stoi(str.substr(0, 1));
	y = stoi(str.substr(2, 1));

	cl::Platform default_platform = all_platforms[x - 1];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[y - 1];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	auto filename =

#if STATE == MAGICK
		"d:\\kernels.cl"
#elif STATE == NOCACHE
		"nocache.cl"
#elif STATE == IMAGE2DT
		"image2d_t.cl"
#elif STATE == IMAGE2DT2
		"image2d_t2.cl"
#elif STATE == SUBGROUP
		"subgroup.cl"
#else 
		"test.cl"
#endif

		;

	std::ifstream stream(filename);

/*#ifndef MAGICK
	std::ifstream stream("main.cl");
#else
	std::ifstream stream("d:\\kernels.cl");
#endif*/
	std::string code((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());


	cl::Context context({ default_device });
	cl::Program::Sources sources;
	sources.push_back({ code.c_str(), code.length() });

	cl::Program program(context, sources);
	auto opts =
#if STATE == NOCACHE || STATE == IMAGE2DT2
		"-cl-std=CL1.2 -DCLQuantum=float -DCLSignedQuantum=float -DCLPixelType=float4 -DQuantumRange=65535.000000f"
#elif STATE == MAGICK
		"-cl-single-precision-constant -cl-mad-enable -DMAGICKCORE_HDRI_SUPPORT=1 -DCLQuantum=float -DCLSignedQuantum=float -DCLPixelType=float4 -DQuantumRange=65535.000000f -DQuantumScale=0.000015 -DCharQuantumScale=1.000000 -DMagickEpsilon=0.000000 -DMagickPI=3.141593 -DMaxMap=65535 -DMAGICKCORE_QUANTUM_DEPTH=16"
#elif STATE == SUBGROUP
		"-cl-std=CL1.2 -DCLQuantum=float -DCLSignedQuantum=float -DCLPixelType=float4 -DQuantumRange=65535.000000f"
#else
		""
#endif
		;
	if (program.build({ default_device }, opts) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

#if STATE == TEST

	float b[16 * 4 * 2];
	for (int i = 0; i < 16 * 4 *2; ++i) b[i] = i;
	cl_int res = 0;

	cl::Buffer input(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 16 * 2 * sizeof(cl_float4), b, &res);

	cl::CommandQueue queue(context, default_device, FLAG, &res);
	cl::Kernel test(program, "test");
	res = test.setArg(0, input);

	res = queue.enqueueNDRangeKernel(test, cl::NullRange, cl::NDRange(32), cl::NullRange);
	res = queue.finish();


#else

	FILE *fp; fopen_s(&fp, argv[1], "rb");
	char header[8];
	fread(header, 1, 8, fp);
	auto png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	auto info_ptr = png_create_info_struct(png_ptr);
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	png_read_info(png_ptr, info_ptr);
	auto width = png_get_image_width(png_ptr, info_ptr);
	auto height = png_get_image_height(png_ptr, info_ptr);
	png_read_update_info(png_ptr, info_ptr);
	setjmp(png_jmpbuf(png_ptr));
	auto raw_width = png_get_rowbytes(png_ptr, info_ptr);
	auto rows = new png_bytep[height];
	for (int i = 0; i < height; ++i) rows[i] = new png_byte[raw_width];
	png_read_image(png_ptr, rows);
	std::vector<float> pixels(raw_width*height);
	for (unsigned i = 0; i < height; ++i) {
		for (unsigned j = 0; j < raw_width; ++j) {
			pixels[i*raw_width + j] = rows[i][j];	
		}
		//delete[] rows[i];
	}
	//delete[] rows;
	fclose(fp);

	auto radius = std::stof(argv[2], nullptr);
	auto sigma = std::stof(argv[3], nullptr);
	auto local_size = std::stol(argv[4], nullptr, 10);

	auto kernel = GenerateKernel(radius, sigma);
	std::cout << "kernel(" << kernel.size() << "):";
	for (auto & k : kernel) std::cout << k << ";";
	std::cout << std::endl;
	cl_int res = 0;
#if STATE == IMAGE2DT || STATE == IMAGE2DT2 || STATE == SUBGROUP
	cl::ImageFormat format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_RGBA;
#endif

#if STATE == IMAGE2DT
	cl::Image2D input(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, format, width, height, 0, pixels.data(), &res);
#else
	cl::Buffer input(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, raw_width * height * sizeof(float), pixels.data(), &res);
#endif

#if STATE == NOCACHE || STATE == MAGICK
	cl::Buffer temp(context, CL_MEM_READ_WRITE, raw_width * height * sizeof(float), nullptr, &res);
#else
	cl::Image2D temp(context, CL_MEM_READ_WRITE, format, width, height, 0, nullptr, &res);
#endif

#if STATE == IMAGE2DT
	cl::Image2D output(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, width, height, 0, nullptr, &res);
#else
	cl::Buffer output(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, raw_width * height * sizeof(float));
#endif

	cl::Buffer kern(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, kernel.size() * sizeof(float), kernel.data(), &res);

	
	cl::CommandQueue queue(context, default_device, FLAG, &res);
	cl::Kernel blur_row(program, "BlurRow");
	int k = 0;
	res = blur_row.setArg(k++, input);
#if STATE != IMAGE2DT
	res = blur_row.setArg(k++, 3U);
	res = blur_row.setArg(k++, 7);
	res = blur_row.setArg(k++, kern);
	res = blur_row.setArg(k++, static_cast<unsigned>(kernel.size()));
	res = blur_row.setArg(k++, width);
	res = blur_row.setArg(k++, height);
#if STATE != NOCACHE
	res = blur_row.setArg(k++, sizeof(cl_float4) * (kernel.size() + local_size), nullptr);
#endif
	res = blur_row.setArg(k++, temp);
#else
	res = blur_row.setArg(k++, temp);
	res = blur_row.setArg(k++, static_cast<unsigned>(kernel.size()));
	res = blur_row.setArg(k++, kern);
	res = blur_row.setArg(k++, width);
#endif
	auto glob = cl::NDRange(local_size*((width + local_size - 1) / local_size), height);
#if PROFILING == 1
	cl::Event evt;
	res = queue.enqueueNDRangeKernel(blur_row, cl::NullRange, glob, cl::NDRange(local_size, 1), nullptr, &evt);
	res = queue.finish();
	std::cout << "[row]time us: " << 1.0e-9 * (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
#else
	res = queue.enqueueNDRangeKernel(blur_row, cl::NullRange, glob, cl::NDRange(local_size, 1));
	res = queue.finish();
#endif


	cl::Kernel blur_col(program, "BlurColumn");
	k = 0;
	res = blur_col.setArg(k++, temp);
#if STATE == NOCACHE || STATE == MAGICK
	res = blur_col.setArg(k++, 3U);
	res = blur_col.setArg(k++, 0x7);
	res = blur_col.setArg(k++, kern);
	res = blur_col.setArg(k++, static_cast<unsigned>(kernel.size()));
	res = blur_col.setArg(k++, width);
	res = blur_col.setArg(k++, height);
#if STATE != NOCACHE
	res = blur_col.setArg(k++, sizeof(cl_float4) * (kernel.size() + local_size), nullptr);
#endif
	res = blur_col.setArg(k++, output);
#elif STATE == IMAGE2DT || STATE == IMAGE2DT2 || STATE == SUBGROUP
	res = blur_col.setArg(k++, output);
	res = blur_col.setArg(k++, static_cast<unsigned>(kernel.size()));
	res = blur_col.setArg(k++, kern);
	res = blur_col.setArg(k++, height);
	res = blur_col.setArg(k++, width);
#if STATE == IMAGE2DT2 || STATE == SUBGROUP
	res = blur_col.setArg(k++, 3U);
	res = blur_col.setArg(k++, 7);
#endif
#endif
	glob = cl::NDRange(width, local_size*((height + local_size - 1) / local_size));
#if PROFILING == 1
	res = queue.enqueueNDRangeKernel(blur_col, cl::NullRange, glob, cl::NDRange(1, local_size), nullptr, &evt);
	res = queue.finish();
	std::cout << "[column]time us: " << 1.0e-9 * (evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << std::endl;
#else
	res = queue.enqueueNDRangeKernel(blur_col, cl::NullRange, glob, cl::NDRange(1, local_size));
	res = queue.finish();
#endif
#ifdef WRITEIMAGE
	std::vector<float> tmp(raw_width * height);
#if STATE != IMAGE2DT
	res = queue.enqueueReadBuffer(output, CL_TRUE, 0, raw_width*height * sizeof(float), tmp.data());
#else
	res = queue.enqueueReadImage(output, CL_TRUE, GetSizeT(0, 0, 0), GetSizeT(width, height, 1), sizeof(cl_float) * raw_width, 0, tmp.data());
#endif
	fopen_s(&fp, argv[5], "wb");
	png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	info_ptr = png_create_info_struct(png_ptr);
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height,
		8, 
#if IMAGE_FORMAT == RGB
		PNG_COLOR_TYPE_RGB
#else
		PNG_COLOR_TYPE_RGBA
#endif
		, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png_ptr, info_ptr);
	for (unsigned i = 0; i < height; ++i) {
		for (unsigned j = 0; j < raw_width; ++j) {
			rows[i][j] = tmp[i*raw_width + j];
		}
		png_write_row(png_ptr, rows[i]);
		delete[] rows[i];
	}
	png_write_end(png_ptr, info_ptr);
	delete[] rows;
	fclose(fp);
#endif
#endif

    return 0;
}

