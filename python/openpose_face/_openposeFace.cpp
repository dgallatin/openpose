#ifndef OPENPOSE_FACE_PYTHON_HPP
#define OPENPOSE_FACE_PYTHON_HPP
#define BOOST_DATE_TIME_NO_LIB

// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <caffe/caffe.hpp>
#include <stdlib.h>

#include <openpose/face/faceExtractorNet.hpp>
#include <openpose/gpu/cuda.hpp>
#include <openpose/gpu/opencl.hcl>
#include <openpose/core/macros.hpp>

#ifdef _WIN32
    #define OP_EXPORT __declspec(dllexport)
#else
    #define OP_EXPORT
#endif

#define default_logging_level 3
#define default_net_input_size "90x90"
#define default_net_output_size "90x90"
#define default_num_gpu_start 0
#define default_model_folder "models/"

// Todo, have GPU Number, handle, OpenCL/CPU Cases
OP_API class OpenPoseFace {
public:
	std::unique_ptr<op::FaceExtractorCaffe> faceExtractorCaffe;
	int mGpuID;

	OpenPoseFace(int FLAGS_logging_level = default_logging_level,
			     const std::string& FLAGS_net_input_size = default_net_input_size,
			     const std::string& FLAGS_net_output_size = default_net_output_size,
                 const std::string& FLAGS_model_folder = default_model_folder,
			     const int FLAGS_num_gpu_start = default_num_gpu_start)
	{
		mGpuID = FLAGS_num_gpu_start;
#ifdef USE_CUDA
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		caffe::Caffe::SetDevice(mGpuID);
#elif USE_OPENCL
		caffe::Caffe::set_mode(caffe::Caffe::GPU);
		std::vector<int> devices;
		const int maxNumberGpu = op::OpenCL::getTotalGPU();
		for (auto i = 0; i < maxNumberGpu; i++)
			devices.emplace_back(i);
		caffe::Caffe::SetDevices(devices);
		caffe::Caffe::SelectDevice(mGpuID, true);
		op::OpenCL::getInstance(mGpuID, CL_DEVICE_TYPE_GPU, true);
#else
		caffe::Caffe::set_mode(caffe::Caffe::CPU);
#endif
		op::log("OpenPoseFace Library Python Wrapper", op::Priority::High);
		// ------------------------- INITIALIZATION -------------------------
		// Step 1 - Set logging level
		// - 0 will output all the logging messages
		// - 255 will output nothing
		op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
		op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
		// Step 2 - Read Google flags (user defined configuration)
		// outputSize
		const auto outputSize = op::flagsToPoint(FLAGS_net_output_size, "256x256");
		// netInputSize
		const auto netInputSize = op::flagsToPoint(FLAGS_net_input_size, "256x256");

		// Step 3 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
		faceExtractorCaffe = std::unique_ptr<op::FaceExtractorCaffe>(new op::FaceExtractorCaffe{ netInputSize, outputSize, FLAGS_model_folder, mGpuID });
		faceExtractorCaffe->initializationOnThread();
	}

	void forward(const cv::Mat& inputImage, std::vector<op::Rectangle<float>>& faceRectangles, op::Array<float>& faceKeypoints) {
		op::OpOutputToCvMat opOutputToCvMat;
		op::CvMatToOpInput cvMatToOpInput;
		op::CvMatToOpOutput cvMatToOpOutput;
		if (inputImage.empty())
			op::error("Could not open or find the image: ", __LINE__, __FUNCTION__, __FILE__);
		const op::Point<int> imageSize{ inputImage.cols, inputImage.rows };

		// Estimate faceKeypoints
		faceExtractorCaffe->forwardPass(faceRectangles, inputImage);
		faceKeypoints = faceExtractorCaffe->getFaceKeypoints();
	}
};

#ifdef __cplusplus
extern "C" {
#endif

	struct rect {
		float x, y, width, height;
	};

	typedef void* c_OP;
	op::Array<float> output;

    OP_EXPORT c_OP newOPFace(int logging_level,
		char* output_resolution,
		char* net_resolution,
		int num_gpu_start,
		char* model_folder
	) {
		return new OpenPoseFace(logging_level, output_resolution, net_resolution, model_folder, num_gpu_start);
	}
    OP_EXPORT void delOPFace(c_OP op) {
		delete (OpenPoseFace *)op;
	}
    OP_EXPORT void forward(c_OP op, unsigned char* img, size_t rows, size_t cols, rect* rectangles, int numFaces, int* size) {
		OpenPoseFace* openPoseFace = (OpenPoseFace*)op;
		cv::Mat image(rows, cols, CV_8UC3, img);
		std::vector<op::Rectangle<float>> faceRectangles;
		for(int i=0; i < numFaces; i++){
			rect curr_dat = rectangles[i];
			op::Rectangle<float> *curr = new op::Rectangle<float> { curr_dat.x, curr_dat.y, curr_dat.width, curr_dat.height };
			faceRectangles.push_back(*curr);
		}

		openPoseFace->forward(image, faceRectangles, output);
		if (output.getSize().size()) {
			size[0] = output.getSize()[0];
			size[1] = output.getSize()[1];
			size[2] = output.getSize()[2];
		}
		else {
			size[0] = 0; size[1] = 0; size[2] = 0;
		}
	}
    OP_EXPORT void getOutputs(c_OP op, float* array) {
		if (output.getSize().size())
			memcpy(array, output.getPtr(), output.getSize()[0] * output.getSize()[1] * output.getSize()[2] * sizeof(float));
	}

#ifdef __cplusplus
}
#endif

#endif
