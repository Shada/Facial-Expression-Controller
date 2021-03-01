// SimpleCLM.cpp : Defines the entry point for the console application.
#include "CLM_core.h"

#include <fstream>
#include <sstream>
#include <ctime>

#include <opencv2\videoio\videoio.hpp>

#include "vJoyFeeder.h"

#define FACE_INPUT 1
#define HEAD_INPUT 2

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

bool read_input_configuration(char* filename);

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
	abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
using namespace cv;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

vJoyFeeder *joyFeeder; // virtual joystick

static int INPUT_METHOD = FACE_INPUT; // current input method

enum tobievent
{
	te_left_mouth_corner_stretch = 0,
	te_right_mouth_corner_stretch,
	te_stretch_mouth, // "smile"
	te_pucker, // "kiss mouth"
	te_wink_left,
	te_wink_right,
	te_blink,
	te_move_mouth_left, // move the whole mouth to the left
	te_move_mouth_right,
	te_mouth_open,
	te_eyebrows_lift,
	te_roll,
	te_pitch,
	te_yaw,
	te_none,
	te_numeventnames,
};

char* te_names[] =
{
	"left_mouth_corner_stretch",
	"right_mouth_corner_stretch",
	"stretch_mouth", // "smile"
	"pucker", // "kiss mouth"
	"wink_left",
	"wink_right",
	"blink",
	"move_mouth_left", // move the whole mouth to the left
	"move_mouth_right",
	"mouth_open",
	"eyebrows_lift",
	"roll",
	"pitch",
	"yaw",
	"none",
};

struct tobibutton
{
	int id; // joystick button_id
	tobievent face_head_event; // name of the event that will trigger the event. distance/ratio
	double threshold; // the value for which the event is triggered
	bool morethan; // default true. Sometimes we want to be below the threshhold to trigger event (pucker mouth)
	int inputtype;
};

std::vector<tobibutton> triggers;
// output log files
std::ofstream pose_output_file;
std::ofstream landmarks_output_file;
std::ofstream distances_output_file;
std::ofstream ratios_output_file;

bool video_output = false;
bool log_output = false;

int log_id = 0;
int video_id = 0;
string session_start_time_string = "";

// Visualising the results
void visualise_tracking(Mat& captured_image, const CLMTracker::CLM& clm_model, Vec6d &pose, const CLMTracker::CLMParameters& clm_parameters, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = clm_model.detection_certainty;
	bool detection_success = clm_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		CLMTracker::Draw(captured_image, clm_model);

		//double vis_certainty = detection_certainty;
		//if (vis_certainty > 1)
		//	vis_certainty = 1;
		//if (vis_certainty < -1)
		//	vis_certainty = -1;
		//
		//vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);
		//
		//// A rough heuristic for box around the face width
		//int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);
		//
		//// Draw it in reddish if uncertain, blueish if certain
		//CLMTracker::DrawBox(captured_image, pose, Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 255));
}

struct face_distances
{
	double mouth_width; // distance between mouth corners
	double mouth_inner_height; // distance between lip, inner midpoints
	double mouth_outer_height; // distance between lip, puter midpoints
	double mouth_center_to_right_corner; // distance between center of lip to right corner
	double mouth_center_to_left_corner; // distance between center of lip to left corner
	double left_mouth_corner_middle_face; // distance between left mouth corner and left face side
	double right_mouth_corner_middle_face; // distance between right mouth corner and right face side
	double left_eye_width; // width of left eye
	double left_eye_height; // height of left eye
	double right_eye_width; // width of right eye
	double right_eye_height; // height of right eye
	double nose_length; // length of nose (used ot detect if looking down)
	double left_eyebrow_left_eye_distance;  // height of left eyebrow relative to eye
	double right_eyebrow_right_eye_distance;// height of right eyebrow relative to eye
	double eye_distance; // distance between eyes
	double face_width; // width of face;
	double face_height; // height of face;
	double nosetip_to_face_left; // nosetip to face left;
	double nosetip_to_face_right; // nosetip to face right;
	double belownose_to_mouth_right;
	double belownose_to_mouth_left;

	double mouth_average_pos_x;
};

struct face_ratios
{
	double left_eye_ratio; // left_eye_height / left_eye_width
	double right_eye_ratio; // right_eye_height / right_eye_width
	double inner_mouth_ratio; // mouth_inner_height / mouth_width
	double outer_mouth_ratio; // mouth_outer_height / mouth_width
	double nose_length_eyewidth_ratio; // nose_length / eye_distance
	double mouth_eye_ratio; // mouth_width / eye_distance
	double mouth_left_right_ratio; // left_mouth_corner_middle_face / right_mouth_corner_middle_face
	double left_eyebrow_height_eye_width_ratio;
	double right_eyebrow_height_eye_width_ratio;
	double left_right_mouth_nose;
};

face_distances fd_current_frame;
face_distances fd_prev_frame;
face_distances fd_first_frame;

face_ratios fr_current_frame;
face_ratios fr_prev_frame;
face_ratios fr_first_frame;

static bool initializedistances = false;


void calc_ratios(face_distances &distances, face_ratios &ratios)
{
	ratios.left_eye_ratio = distances.left_eye_height / distances.left_eye_width;
	ratios.right_eye_ratio = distances.right_eye_height / distances.right_eye_width;
	ratios.inner_mouth_ratio = distances.mouth_inner_height / distances.face_width;
	ratios.outer_mouth_ratio = distances.mouth_outer_height / distances.mouth_width;
	ratios.nose_length_eyewidth_ratio = distances.nose_length / distances.eye_distance;
	ratios.mouth_eye_ratio = distances.mouth_width / distances.eye_distance;
	ratios.mouth_left_right_ratio = distances.left_mouth_corner_middle_face / distances.right_mouth_corner_middle_face;
	ratios.left_eyebrow_height_eye_width_ratio = distances.left_eyebrow_left_eye_distance / distances.eye_distance;
	ratios.right_eyebrow_height_eye_width_ratio = distances.right_eyebrow_right_eye_distance / distances.eye_distance;
	ratios.left_right_mouth_nose = distances.belownose_to_mouth_left / distances.belownose_to_mouth_right;
}

void calc_distances(face_distances &distances, const CLMTracker::CLM &clm_model)
{
	
	double x, y;
	int n = clm_model.detected_landmarks.rows / 2;
	//fd_current_frame.mouth_width = std::sqrt()


	//**--**--**--**--***--**--**--**--**//
	//**--**--**--** MOUTH **--**--**--**//
	//**--**--**--**--***--**--**--**--**//

	// mouth inner height
	x = (clm_model.detected_landmarks.at<double>(62) - clm_model.detected_landmarks.at<double>(66));
	y = (clm_model.detected_landmarks.at<double>(62 + n) - clm_model.detected_landmarks.at<double>(66 + n));
	distances.mouth_inner_height = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	// debug info on screen

	// mouth outer height
	x = (clm_model.detected_landmarks.at<double>(51) - clm_model.detected_landmarks.at<double>(57));
	y = (clm_model.detected_landmarks.at<double>(51 + n) - clm_model.detected_landmarks.at<double>(57 + n));
	distances.mouth_outer_height = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	// mouth width
	x = (clm_model.detected_landmarks.at<double>(48) - clm_model.detected_landmarks.at<double>(54));
	y = (clm_model.detected_landmarks.at<double>(48 + n) - clm_model.detected_landmarks.at<double>(54 + n));
	distances.mouth_width = std::sqrt(x*x + y*y) / clm_model.params_global[0];


	x = (clm_model.detected_landmarks.at<double>(48) - clm_model.detected_landmarks.at<double>(3));
	distances.left_mouth_corner_middle_face = std::abs(x) / clm_model.params_global[0];


	x = (clm_model.detected_landmarks.at<double>(54) - clm_model.detected_landmarks.at<double>(13));
	distances.right_mouth_corner_middle_face = std::abs(x) / clm_model.params_global[0];

	//**--**--**--**--**--**--**--**--**//
	//**--**--**--** EYES **--**--**--**//
	//**--**--**--**--**--**--**--**--**//

	// left eye height
	x = (clm_model.detected_landmarks.at<double>(37) - clm_model.detected_landmarks.at<double>(41));
	y = (clm_model.detected_landmarks.at<double>(37 + n) - clm_model.detected_landmarks.at<double>(41 + n));
	distances.left_eye_height = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	// left eye width
	x = (clm_model.detected_landmarks.at<double>(36) - clm_model.detected_landmarks.at<double>(39));
	y = (clm_model.detected_landmarks.at<double>(36 + n) - clm_model.detected_landmarks.at<double>(39 + n));
	distances.left_eye_width = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	// right eye height
	x = (clm_model.detected_landmarks.at<double>(44) - clm_model.detected_landmarks.at<double>(46));
	y = (clm_model.detected_landmarks.at<double>(44 + n) - clm_model.detected_landmarks.at<double>(46 + n));
	distances.right_eye_height = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	//right eye width
	x = (clm_model.detected_landmarks.at<double>(42) - clm_model.detected_landmarks.at<double>(45));
	y = (clm_model.detected_landmarks.at<double>(42 + n) - clm_model.detected_landmarks.at<double>(45 + n));
	distances.right_eye_width = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	//eye distance ( inner corners )
	x = (clm_model.detected_landmarks.at<double>(39) - clm_model.detected_landmarks.at<double>(42));
	y = (clm_model.detected_landmarks.at<double>(39 + n) - clm_model.detected_landmarks.at<double>(42 + n));
	distances.eye_distance = std::sqrt(x*x + y*y) / clm_model.params_global[0];


	//**--**--**--**--**--**--**--**--**//
	//**--**--**--** NOSE **--**--**--**//
	//**--**--**--**--**--**--**--**--**//

	x = (clm_model.detected_landmarks.at<double>(27) - clm_model.detected_landmarks.at<double>(30));
	y = (clm_model.detected_landmarks.at<double>(27 + n) - clm_model.detected_landmarks.at<double>(30 + n));
	distances.nose_length = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	//**--**--**--**--***--**--**--**--**//
	//**--**--**--** OTHER **--**--**--**//
	//**--**--**--**--***--**--**--**--**//

	x = (clm_model.detected_landmarks.at<double>(0) - clm_model.detected_landmarks.at<double>(16));
	y = (clm_model.detected_landmarks.at<double>(0 + n) - clm_model.detected_landmarks.at<double>(16 + n));
	distances.face_width = std::sqrt(x*x + y*y) / clm_model.params_global[0];

	y = (clm_model.detected_landmarks.at<double>(19 + n) - clm_model.detected_landmarks.at<double>(39 + n));
	distances.left_eyebrow_left_eye_distance = std::abs(y) / clm_model.params_global[0];

	y = (clm_model.detected_landmarks.at<double>(22 + n) - clm_model.detected_landmarks.at<double>(42 + n));
	distances.right_eyebrow_right_eye_distance = std::abs(y) / clm_model.params_global[0];

	double eye_center = clm_model.detected_landmarks.at<double>(36) + (clm_model.detected_landmarks.at<double>(45) - clm_model.detected_landmarks.at<double>(36)) / 2;
	x = /*(clm_model.detected_landmarks.at<double>(45)*/ eye_center - clm_model.detected_landmarks.at<double>(54);
	//y = (clm_model.detected_landmarks.at<double>(33 + n) - clm_model.detected_landmarks.at<double>(54 + n));
	distances.belownose_to_mouth_right = std::abs(x) / clm_model.params_global[0];

	x = (/*clm_model.detected_landmarks.at<double>(36)*/ eye_center - clm_model.detected_landmarks.at<double>(48));
	//y = (clm_model.detected_landmarks.at<double>(33 + n) - clm_model.detected_landmarks.at<double>(48 + n));
	distances.belownose_to_mouth_left = std::abs(x) / clm_model.params_global[0];

	distances.mouth_average_pos_x = ((clm_model.detected_landmarks.at<double>(48) + clm_model.detected_landmarks.at<double>(54) +
		clm_model.detected_landmarks.at<double>(62) + clm_model.detected_landmarks.at<double>(66) - clm_model.params_global[4] * 4) / 4) / clm_model.params_global[0];
}

void interpretlandmarkevents(Vec6d &pose)
{

	//joyFeeder->setWAxisX(45 + pose[4] * 180.0 / M_PI);
	//joyFeeder->setWAxisY(45 + pose[3] * 180.0 / M_PI);
	//joyFeeder->setWAxisZ(45 + pose[5] * 180.0 / M_PI);

	fd_prev_frame = fd_current_frame;
	for (int i = 0; i < 8; i++)
	{
		joyFeeder->setBtn(i, false);
	}

	for each(tobibutton tb in triggers)
	{
		if (tb.inputtype == INPUT_METHOD)
		{
			switch (tb.face_head_event)
			{
			case tobievent::te_mouth_open:
			{
				// check if mouth is open, move to some other place maybe.. some file where I have everything with the interpretations..
				if (fd_current_frame.mouth_outer_height > fd_first_frame.mouth_outer_height + tb.threshold)
				{
					joyFeeder->setBtn(tb.id, true);
				}
				break;
			}
			case tobievent::te_move_mouth_left:
			{
				if (fd_current_frame.belownose_to_mouth_left > fd_first_frame.belownose_to_mouth_left + tb.threshold)
				{
					joyFeeder->setBtn(tb.id, true);
				}
				break;
			}
			case tobievent::te_move_mouth_right:
			{
				if (fd_current_frame.belownose_to_mouth_right > fd_first_frame.belownose_to_mouth_right + tb.threshold)
				{
					joyFeeder->setBtn(tb.id, true);
				}
				break;
			}
			case tobievent::te_left_mouth_corner_stretch:
				break;
			case tobievent::te_right_mouth_corner_stretch:
				break;
			case tobievent::te_stretch_mouth:
				break;
			case tobievent::te_pucker:
				break;
			case tobievent::te_wink_left:
				break;
			case tobievent::te_wink_right:
				break;
			case tobievent::te_blink:
				break;
			case tobievent::te_eyebrows_lift:
				if(fd_current_frame.left_eyebrow_left_eye_distance > fd_first_frame.left_eyebrow_left_eye_distance + 2)
					joyFeeder->setBtn(tb.id, true);
				break;
			case tobievent::te_none: // this can be used when the set event does not exist.
				break;
			case tobievent::te_roll:
			{
				if (tb.morethan)
				{
					if (pose[5] > tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				else
				{
					if (pose[5] < tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				break;
			}
			case tobievent::te_yaw:
			{
				if (tb.morethan)
				{
					if (pose[4] > tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				else
				{
					if (pose[4] < tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				break;
			}
			case tobievent::te_pitch:
			{
				if (tb.morethan)
				{
					if (pose[3] > tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				else
				{
					if (pose[3] < tb.threshold)
					{
						joyFeeder->setBtn(tb.id, true);
					}
				}
				break;
			}
			default:  
				break;
			}
		}
	}
}

int main(int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Some initial parameters that can be overriden from command line	
	string input_video_filename, output_video_filename, pose_output_filename, landmark_output_filename, distances_output_filename, ratios_output_filename;

	// By default try webcam 0
	int device = 0;

	// virtual joystick
	joyFeeder = new vJoyFeeder();

	if (!joyFeeder->initialize(1))
		std::cout << "Failed to initialize JoyFeeder" << std::endl;
	else
	{
		for (int i = 0; i < 8; i++)
			joyFeeder->setBtn(i, false);

		joyFeeder->setWAxisX(45);
		joyFeeder->setWAxisY(45);
		joyFeeder->setWAxisZ(45);
		joyFeeder->sendMessage();

	}
	// clm-framework
	CLMTracker::CLMParameters clm_parameters(arguments);

	// Get the input output file parameters

	// Indicates that rotation should be with respect to world or camera coordinates
	bool use_world_coordinates;
	CLMTracker::get_video_input_output_params(input_video_filename, use_world_coordinates, arguments);

	// The modules that are being used for tracking
	CLMTracker::CLM clm_model(clm_parameters.model_location);

	if (clm_parameters.curr_face_detector == 0)
		clm_model.face_detector_HAAR = cv::CascadeClassifier("./classifiers/haarcascade_frontalface_alt.xml");

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	// Get camera parameters
	CLMTracker::get_camera_params(device, fx, fy, cx, cy, arguments);

	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// If multiple video files are tracked, use this to indicate if we are done

	double fps_vid_in = -1.0;

	// Do some grabbing
	VideoCapture video_capture;
	VideoWriter video_writer;
	VideoWriter video_writer2;
	if (!input_video_filename.empty()) // video file instead of webcamera
	{
		INFO_STREAM("Attempting to read from file: " << input_video_filename);
		video_capture = VideoCapture(input_video_filename);
		fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

		// Check if fps is nan or less than 0
		if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
		{
			INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
			fps_vid_in = 30;
		}
	}
	else // webcamera
	{
		INFO_STREAM("Attempting to capture from device: " << device);
		video_capture = VideoCapture(device);

		// Read a first frame often empty in camera
		Mat captured_image;
		video_capture >> captured_image;
	}

	if (!video_capture.isOpened()) FATAL_STREAM("Failed to open video source");
	else INFO_STREAM("Device or file opened");

	Mat captured_image;
	video_capture >> captured_image;

	// If optical centers are not defined just use center of image
	if (cx_undefined)
	{
		cx = captured_image.cols / 2.0f;
		cy = captured_image.rows / 2.0f;
	}
	// Use a rough guess-timate of focal length
	if (fx_undefined)
	{
		fx = 500 * (captured_image.cols / 640.0);
		fy = 500 * (captured_image.rows / 480.0);

		fx = (fx + fy) / 2.0;
		fy = fx;
	}


	int frame_count = 0;

	// Use for timestamping if using a webcam
	int64 t_initial = cv::getTickCount();

	// Timestamp in seconds of current processing
	double time_stamp = 0;

	INFO_STREAM("Starting tracking");
	Mat_<uchar> grayscale_image;
	Mat_<float> empty;

	bool video_output = false;
	bool log_output = false;
	bool done = false;

	int video_frame = 0;
	int current_video = 0;

	clm_parameters.quiet_mode = false;

	time_t t = time(0);
	struct tm * now = localtime(&t);
	session_start_time_string = to_string(now->tm_year + 1900) + "-" + to_string(now->tm_mon + 1) + "-" + to_string(now->tm_mday) + "-" + to_string(now->tm_hour) + "-" + to_string(now->tm_min) + "-" + to_string(now->tm_sec);
	int log_id = 0;

	if (!read_input_configuration("TOBIINPUT.txt"))
	{
		cout << "COULDN'T READ FILE: TOBIINPUT.txt" << endl;
	}

	while (!done && !captured_image.empty())
	{

		// Grab the timestamp first
		if (fps_vid_in == -1)
		{
			int64 curr_time = cv::getTickCount();
			time_stamp = (double(curr_time - t_initial) / cv::getTickFrequency());
		}
		else
		{
			time_stamp = (double)frame_count * (1.0 / fps_vid_in);
		}

		// Reading the images
		if (captured_image.channels() == 3)
		{
			cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
		}
		else
		{
			grayscale_image = captured_image.clone();
		}

		// The actual facial landmark detection / tracking
		bool detection_success = CLMTracker::DetectLandmarksInVideo(grayscale_image, clm_model, clm_parameters);

		// Work out the pose of the head from the tracked model
		Vec6d pose_estimate_CLM;
		if (use_world_coordinates)
		{
			pose_estimate_CLM = CLMTracker::GetCorrectedPoseWorld(clm_model, fx, fy, cx, cy);
		}
		else
		{
			pose_estimate_CLM = CLMTracker::GetCorrectedPoseCamera(clm_model, fx, fy, cx, cy);
		}

		// check if we have initialized distances
		if (initializedistances == false)
		{
			if (detection_success)
			{
				calc_distances(fd_first_frame, clm_model);
				calc_ratios(fd_first_frame, fr_first_frame);
				initializedistances = true;
			}
		}
		
		if (detection_success)
		{
			// calculate distances and ratios for current frame
			calc_distances(fd_current_frame, clm_model);
			calc_ratios(fd_current_frame, fr_current_frame);
		}

		if (detection_success)
		{
			interpretlandmarkevents(pose_estimate_CLM);
			joyFeeder->sendMessage(); // joystick send message (TODO: check if it is the same message... if it is the same, we don't need to send again??, or iif it is an empty message)

		}

		// write the raw video
		if (video_output) // no overlay
		{
			video_writer2 << captured_image;
			video_frame++;
		}


		double detection_certainty = clm_model.detection_certainty;

		// Visualising the results
		// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	
		visualise_tracking(captured_image, clm_model, pose_estimate_CLM, clm_parameters, frame_count, fx, fy, cx, cy);
		if (!clm_parameters.quiet_mode)
		{
			imshow("tobiTracker", captured_image);
		}
		// write video with overlay
		if (video_output) // with overlay
			video_writer << captured_image;



		// Output the detected facial landmarks
		if (log_output)
		{
			double confidence = 0.5 * (1 - clm_model.detection_certainty);

			// landmark log
			landmarks_output_file << frame_count + 1 << ", " << time_stamp << ", " << current_video << ", " << video_frame << ", " << confidence << ", " << detection_success;
			for (int i = 0; i < clm_model.pdm.NumberOfPoints() * 2; ++i)
			{
				landmarks_output_file << ", " << clm_model.detected_landmarks.at<double>(i);
			}
			landmarks_output_file << endl;

			// pose log
			pose_output_file << frame_count + 1 << ", " << time_stamp << ", " << current_video << ", " << video_frame << ", " << confidence << ", " << detection_success
				<< ", " << pose_estimate_CLM[0] << ", " << pose_estimate_CLM[1] << ", " << pose_estimate_CLM[2]
				<< ", " << pose_estimate_CLM[3] << ", " << pose_estimate_CLM[4] << ", " << pose_estimate_CLM[5] << endl;

			// distances log
			distances_output_file << frame_count + 1 << ", " << time_stamp << ", " << current_video << ", " << video_frame << ", " << confidence << ", " << detection_success << ", " << 
				fd_current_frame.mouth_width << ", " << fd_current_frame.mouth_inner_height << ", " << fd_current_frame.mouth_outer_height << ", " << fd_current_frame.mouth_center_to_right_corner << ", " <<
				fd_current_frame.mouth_center_to_left_corner << ", " << fd_current_frame.left_mouth_corner_middle_face << ", " << fd_current_frame.right_mouth_corner_middle_face << ", " << fd_current_frame.left_eye_width << ", " << 
				fd_current_frame.left_eye_height << ", " << fd_current_frame.right_eye_width << ", " << fd_current_frame.right_eye_height << ", " << fd_current_frame.nose_length << ", " << fd_current_frame.left_eyebrow_left_eye_distance << ", " << 
				fd_current_frame.right_eyebrow_right_eye_distance << ", " << fd_current_frame.eye_distance << ", " << fd_current_frame.face_width << ", " << fd_current_frame.face_height << ", " << fd_current_frame.nosetip_to_face_left << ", " << 
				fd_current_frame.nosetip_to_face_right << ", " << fd_current_frame.mouth_average_pos_x  << endl;
			
			//ratios log
			ratios_output_file << frame_count + 1 << ", " << time_stamp << ", " << current_video << ", " << video_frame << ", " << confidence << ", " << detection_success << ", " << 
				fr_current_frame.left_eye_ratio << ", " << fr_current_frame.right_eye_ratio << ", " << fr_current_frame.inner_mouth_ratio << ", " << fr_current_frame.outer_mouth_ratio << ", " << fr_current_frame.nose_length_eyewidth_ratio << ", " <<
				fr_current_frame.mouth_eye_ratio << ", " << fr_current_frame.mouth_left_right_ratio << ", " << fr_current_frame.left_eyebrow_height_eye_width_ratio << ", " << fr_current_frame.right_eyebrow_height_eye_width_ratio << endl;

		}

		// next frame to check
		video_capture >> captured_image;



		// detect key presses
		char character_press = cv::waitKey(1);

		switch (character_press)
		{
			// restart the tracker. This has to be done at the start of each session
		case ('r') :
		{
			clm_model.Reset();
			initializedistances = false;
			video_capture.set(CAP_PROP_POS_FRAMES, 0);
			
			//restart logging: i.e new log file.. 

			//stop the current logging
			pose_output_file.close();
			landmarks_output_file.close();
			distances_output_file.close();
			ratios_output_file.close();

			// start next logging
			// set the filenames
			log_id++;
			string pose_output_filename = "./logs/" + session_start_time_string + "-" + to_string(log_id) + "_pose" + ".txt";
			string landmark_output_filename = "./logs/" + session_start_time_string + "-" + to_string(log_id) + "_landmarks" + ".txt";
			string distances_output_filename = "./logs/" + session_start_time_string + "-" + to_string(log_id) + "_distances" + ".txt";
			string ratios_output_filename = "./logs/" + session_start_time_string + "-" + to_string(log_id) + "_ratios" + ".txt";


			pose_output_file.open(pose_output_filename, ios_base::out);
			pose_output_file << "frame, timestamp, video_id, video_frame, confidence, success, Tx, Ty, Tz, Rx, Ry, Rz";
			pose_output_file << endl;

			landmarks_output_file.open(landmark_output_filename, ios_base::out);
			landmarks_output_file << "frame, timestamp, video_id, video_frame, confidence, success";
			for (int i = 0; i < 68; ++i)
				landmarks_output_file << ", x" << i;

			for (int i = 0; i < 68; ++i)
				landmarks_output_file << ", y" << i;

			landmarks_output_file << endl;

			distances_output_file.open(distances_output_filename, ios_base::out);
			distances_output_file << "frame, timestamp, video_id, video_frame, confidence, success, mouth_width, mouth_inner_height, mouth_outer_height, mouth_center_to_right_corner, mouth_center_to_left_corner, left_mouth_corner_middle_face, " <<
				"right_mouth_corner_middle_face, left_eye_width, left_eye_height, right_eye_width, right_eye_height, nose_length, left_eyebrow_left_eye_distance, right_eyebrow_right_eye_distance, eye_distance, face_width, face_height" <<
				"nosetip_to_face_left, nosetip_to_face_right, mouth_average_x";
			distances_output_file << endl;


			ratios_output_file.open(ratios_output_filename, ios_base::out);
			ratios_output_file << "frame, timestamp, video_id, video_frame, confidence, success, left_eye_ratio, right_eye_ratio, inner_mouth_ratio, outer_mouth_ratio, nose_length_eyewidth_ratio, mouth_eye_ratio, mouth_left_right_ratio, " <<
				"left_eyebrow_height_eye_width_ratio, right_eyebrow_height_eye_width_ratio";
			ratios_output_file << endl;

			log_output = true;

			std::cout << "logging ON" << std::endl;
			break;
		}
		// quit the application
		case ('q') :
		{
			done = true;
			break;
		}
				   // switch to head input
		case ('h') :
		{
			INPUT_METHOD = HEAD_INPUT;
			clm_parameters.curr_face_detector = CLMTracker::CLMParameters::HOG_SVM_DETECTOR;
			break;
		}
		case ('v') : // toggle video ON/OFF. do this any time. preferably at the start/before pressing 'R'. This will start a video recording of raw video and video with landmark overlay
		{

			if (!video_output) // turn on video recording
			{
				string video_output_filename = "./logs/videos/" + session_start_time_string + "-" + to_string(video_id + 1) + ".avi";
				string video_output_filename_overlay = "./logs/videos/" + session_start_time_string + "-" + to_string(video_id + 1) + "_overlay" + ".avi";

				video_writer.open(video_output_filename, CV_FOURCC('D', 'I', 'V', 'X'), 30, captured_image.size(), true);
				video_writer2.open(video_output_filename_overlay, CV_FOURCC('D', 'I', 'V', 'X'), 30, captured_image.size(), true);
				if (video_writer.isOpened() && video_writer2.isOpened())
				{
					video_output = true;
					std::cout << "video recording ON" << std::endl;
					video_id++;
					current_video = video_id;
					video_frame = 0;
				}
			}
			else // turn off video recording
			{
				video_output = false;
				video_writer.release();
				video_writer2.release();
				std::cout << "video recording OFF" << std::endl;
				video_frame = 0;
				current_video = 0;
			}
			break;
		}
		 // switch to face input
		case ('f') :
		{
			INPUT_METHOD = FACE_INPUT;
			clm_parameters.curr_face_detector = CLMTracker::CLMParameters::HAAR_DETECTOR;
			break;
		}
		case 'w':
		{
			clm_parameters.quiet_mode = !clm_parameters.quiet_mode;
			
			break;
		}
		default:
			break;
		}
		// Update the frame count
		frame_count++;

	}

	video_writer.release();
	video_writer2.release();

	pose_output_file.close();
	landmarks_output_file.close();
	distances_output_file.close();
	ratios_output_file.close();

	cv::destroyAllWindows();
	for (int i = 0; i < 8; i++)
		joyFeeder->setBtn(i, false);

	joyFeeder->setWAxisX(45);
	joyFeeder->setWAxisY(45);
	joyFeeder->setWAxisZ(45);
	joyFeeder->sendMessage();
	joyFeeder->shutdown();

	return 0;
}



bool read_input_configuration(char* filename)
{
	cout << "reading configuration file" << endl;
	fstream filestream(filename);

	if (!filestream.is_open())
		return false;

	tobibutton temp;
	int inputtype = FACE_INPUT;
	char c;
	int te_index = 0;
	while (filestream.good())
	{
		c = filestream.peek();
		string line;
		switch (c)
		{
		case 'i': // input type
			// create new input for this input type. 
			filestream >> line;
			filestream >> line;
			if (line == "HEAD_INPUT")
			{
				inputtype = HEAD_INPUT;
			}
			else
			{
				inputtype = FACE_INPUT;
			}
			getline(filestream, line);
			break;
		case 'b':
			// connect button with face/head event and threshold
			filestream >> line;
			filestream >> temp.id;

			filestream >> line;
			
			te_index = 0;
			for each(string name in te_names)
			{
				if (name == line)
				{
					temp.face_head_event = (tobievent)te_index;
					break;
				}
				te_index++;
			}
			
			filestream >> temp.threshold;

			temp.morethan = true;
			if (temp.threshold < 0)
			{
				temp.morethan = false;
			}
			temp.inputtype = inputtype;
			triggers.push_back(temp);
			getline(filestream, line);
			break;
		case '#': // comment
			getline(filestream, line);
			cout << "comment: " << line << endl;
			break;
		default:
			getline(filestream, line);
			break;
		}
	}

	filestream.close();
	return true;
}