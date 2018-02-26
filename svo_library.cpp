/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
#include <svo/global.h>
#include <svo/config.h>
#include <svo/frame_handler_mono.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <sophus/se3.h>

#include <SLAMBenchAPI.h>
#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>


typedef boost::shared_ptr<svo::Frame> FramePtr;


static slambench::io::CameraSensor *grey_sensor;
static vk::AbstractCamera* cam_;
static svo::FrameHandlerMono* vo_;
static cv::Mat *img;
static double img_id = 0.0;
static Sophus::SE3 T_world_from_vision_;
static sb_uint2 inputSize;

//=========================================================================
// SLAMBench output values
//=========================================================================


slambench::outputs::Output *pose_output = nullptr;
slambench::outputs::Output *frame_output = nullptr;
slambench::outputs::Output *pointcloud_output = nullptr;


bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {

    // as a default value use the existing value that was already hard-coded in SVO

    slam_settings->addParameter(TypedParameter<size_t>("p", "pyramid-levels",
        "Number of pyramid levels used for features.",
        &svo::Config::nPyrLevels(), &svo::Config::nPyrLevels()));

    slam_settings->addParameter(TypedParameter<size_t>("ckf", "core-keyframes",
        "Number of keyframes in the core. The core-kfs are optimized through bundle adjustment.",
        &svo::Config::coreNKfs(), &svo::Config::coreNKfs()));

    slam_settings->addParameter(TypedParameter<double>("m", "map-scale",
        "Initial scale of the map. Depends on the distance the camera is moved for the initialization.",
        &svo::Config::mapScale(), &svo::Config::mapScale()));

    slam_settings->addParameter(TypedParameter<size_t>("g", "grid-size",
        "Feature grid size of a cell in [px].",
        &svo::Config::gridSize(), &svo::Config::gridSize()));

    slam_settings->addParameter(TypedParameter<double>("d", "disparity",
        "Initialization: Minimum required disparity between the first two frames.",
        &svo::Config::initMinDisparity(), &svo::Config::initMinDisparity()));

    slam_settings->addParameter(TypedParameter<size_t>("mt", "min-track",
        "Initialization: Minimum number of tracked features.",
        &svo::Config::initMinTracked(), &svo::Config::initMinTracked()));

    slam_settings->addParameter(TypedParameter<size_t>("min", "min-inliers",
        "Initialization: Minimum number of inliers after RANSAC.",
        &svo::Config::initMinInliers(), &svo::Config::initMinInliers()));

    slam_settings->addParameter(TypedParameter<size_t>("Mlk", "max-lucas-kanade",
        "Maximum level of the Lucas Kanade tracker.",
        &svo::Config::kltMaxLevel(), &svo::Config::kltMaxLevel()));

    slam_settings->addParameter(TypedParameter<size_t>("mlk", "min-lucas-kanade",
        "Minimum level of the Lucas Kanade tracker.",
        &svo::Config::kltMinLevel(), &svo::Config::kltMinLevel()));

    slam_settings->addParameter(TypedParameter<double>("rt", "reprojection-threshold",
        "Reprojection threshold [px].",
        &svo::Config::reprojThresh(), &svo::Config::reprojThresh()));

    slam_settings->addParameter(TypedParameter<double>("pot", "pose-optim-threshold",
        "Reprojection threshold after pose optimization.",
        &svo::Config::poseOptimThresh(), &svo::Config::poseOptimThresh()));

    slam_settings->addParameter(TypedParameter<size_t>("poi", "pose-optim-iterations",
        "Number of iterations in local bundle adjustment.",
        &svo::Config::poseOptimNumIter(), &svo::Config::poseOptimNumIter()));

    slam_settings->addParameter(TypedParameter<size_t>("sop", "structure-optim-points",
        "Maximum number of points to optimize at every iteration.",
        &svo::Config::structureOptimMaxPts(), &svo::Config::structureOptimMaxPts()));

    slam_settings->addParameter(TypedParameter<size_t>("soi", "structure-optim-iterations",
        "Number of iterations in structure optimization.",
        &svo::Config::structureOptimNumIter(), &svo::Config::structureOptimNumIter()));

    slam_settings->addParameter(TypedParameter<double>("bat", "ba-threshold",
        "Reprojection threshold after bundle adjustment.",
        &svo::Config::lobaThresh(), &svo::Config::lobaThresh()));

    slam_settings->addParameter(TypedParameter<double>("hbat", "huber-ba-threshold",
        "Threshold for the robust Huber kernel of the local bundle adjustment.",
        &svo::Config::lobaRobustHuberWidth(), &svo::Config::lobaRobustHuberWidth()));

    slam_settings->addParameter(TypedParameter<size_t>("bai", "ba-iterations",
        "Number of iterations in the local bundle adjustment.",
        &svo::Config::lobaNumIter(), &svo::Config::lobaNumIter()));

    slam_settings->addParameter(TypedParameter<double>("dkf", "distance-keyframes",
        "Minimum distance between two keyframes, relative to the average height in the map.",
        &svo::Config::kfSelectMinDist(), &svo::Config::kfSelectMinDist()));

    slam_settings->addParameter(TypedParameter<double>("hcs", "harris-corner-score",
        "Select only features with a minimum Harris corner score for triangulation.",
        &svo::Config::triangMinCornerScore(), &svo::Config::triangMinCornerScore()));

    slam_settings->addParameter(TypedParameter<size_t>("si", "subpixel-iterations",
        "Subpixel refinement of reprojection and triangulation. Set to 0 if no subpix refinement required!",
        &svo::Config::subpixNIter(), &svo::Config::subpixNIter()));

    slam_settings->addParameter(TypedParameter<size_t>("Mkf", "max-keyframes",
        "Limit the number of keyframes in the map (min 3). Set to 0 if unlimited number of keyframes are allowed.",
        &svo::Config::maxNKfs(), &svo::Config::maxNKfs()));

    slam_settings->addParameter(TypedParameter<size_t>("Mft", "max-features-track",
        "Maximum number of features that should be tracked.",
        &svo::Config::maxFts(), &svo::Config::maxFts()));

    slam_settings->addParameter(TypedParameter<size_t>("mft", "min-features-threshold",
        "If the number of tracked features drops below this threshold, tracking quality is bad.",
        &svo::Config::qualityMinFts(), &svo::Config::qualityMinFts()));

    slam_settings->addParameter(TypedParameter<int>("Mftd", "max-features-drop",
        "If within one frame, this amount of features are dropped, tracking quality is bad.",
        &svo::Config::qualityMaxFtsDrop(), &svo::Config::qualityMaxFtsDrop()));

    std::cout << "SVO configured" << std::endl;
    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)  {


 	slambench::io::CameraSensorFinder sensor_finder;
	grey_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "grey"}});


    if (grey_sensor == nullptr) {
        std::cerr << "Invalid sensors found, Grey not found." << std::endl;
        return false;
    }

    if (grey_sensor->PixelFormat != slambench::io::pixelformat::G_I_8) {
        std::cerr << "Grey sensor is not in G_I_8 format" << std::endl;
        return false;
    }

    if (grey_sensor->FrameFormat != slambench::io::frameformat::Raster) {
        std::cerr << "Grey sensor is not in Raster format" << std::endl;
        return false;
    }

    cam_ = new vk::ATANCamera(grey_sensor->Width, grey_sensor->Height,
        grey_sensor->Intrinsics[0], grey_sensor->Intrinsics[1],
        grey_sensor->Intrinsics[2], grey_sensor->Intrinsics[3], 0);
    vo_ = new svo::FrameHandlerMono(cam_);
    vo_->start();
    img = new cv::Mat(grey_sensor->Height, grey_sensor->Width, CV_8UC1);
    inputSize = make_sb_uint2(grey_sensor->Width, grey_sensor->Height);
    Eigen::Matrix3d rotation = grey_sensor->Pose.block(0,0,3,3).cast<double>();
    T_world_from_vision_ = Sophus::SE3(rotation,
        Eigen::Vector3d(grey_sensor->Pose(0,3), grey_sensor->Pose(1,3), grey_sensor->Pose(2,3)));



 	//=========================================================================
 	// DECLARE OUTPTUS
 	//=========================================================================


 	pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
 	slam_settings->GetOutputManager().RegisterOutput(pose_output);
 	pose_output->SetActive(true);

 	frame_output = new slambench::outputs::Output("Frame", slambench::values::VT_FRAME);
 	frame_output->SetKeepOnlyMostRecent(true);
 	slam_settings->GetOutputManager().RegisterOutput(frame_output);
 	frame_output->SetActive(true);

 	 pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_POINTCLOUD);
 	 pointcloud_output->SetKeepOnlyMostRecent(true);
 	 slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);
 	 pointcloud_output->SetActive(true);




    std::cout << "SVO Initialization Successfull." << std::endl;
    return true;

}

bool sb_update_frame (SLAMBenchLibraryHelper * , slambench::io::SLAMFrame* s) {
    if (s->FrameSensor != grey_sensor)
        return false;
    memcpy(img->data, s->GetData(), s->GetSize());
    s->FreeData();
    return true;
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {
    if (img == NULL || img->empty())
        return false;
    img_id += 0.01;
    vo_->addImage(*img, img_id);

    return true;
}

bool sb_get_pose (Eigen::Matrix4f * mat)  {
    const FramePtr& frame = vo_->lastFrame();
    Sophus::SE3 T_world_from_cam(T_world_from_vision_ * frame->T_f_w_.inverse());

    *mat = T_world_from_cam.matrix().cast<float>();

    return true;
}

bool sb_get_tracked  (bool* tracking)  {
    *tracking = (vo_->trackingQuality() ==
        svo::FrameHandlerBase::TrackingQuality::TRACKING_GOOD);

  return true;
}


bool sb_clean_slam_system() {
    delete vo_;
    delete cam_;
    delete img;

    return true;
}

/**
 * GUI sb_initialize_ui
 */


bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output) {
	(void)lib;

	slambench::TimeStamp ts = *latest_output;

	if(pose_output->IsActive()) {

		 const FramePtr& frame = vo_->lastFrame();
		 Sophus::SE3 T_world_from_cam(T_world_from_vision_ * frame->T_f_w_.inverse());

		 Eigen::Matrix4f mat  = T_world_from_cam.matrix().cast<float>();



		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		pose_output->AddPoint(ts, new slambench::values::PoseValue(mat));
	}




		if(frame_output->IsActive()) {

		    const FramePtr& frame = vo_->lastFrame();
		    size_t max_pyr = svo::Config::nPyrLevels();
		    cv::Mat *img_rgb = new cv::Mat(frame->img_pyr_[0].size(), CV_8UC3);
		    cv::cvtColor(frame->img_pyr_[0], *img_rgb, CV_GRAY2RGB);

		    for(svo::Features::iterator it=frame->fts_.begin(); it!=frame->fts_.end(); ++it)
		      {
		        if((*it)->type == svo::Feature::FeatureType::EDGELET)
		          cv::line(*img_rgb,
		                   cv::Point2f((*it)->px[0]+3*(*it)->grad[1], (*it)->px[1]-3*(*it)->grad[0]),
		                   cv::Point2f((*it)->px[0]-3*(*it)->grad[1], (*it)->px[1]+3*(*it)->grad[0]),
		                   cv::Scalar(255,0,255), 2);
		        else
		          cv::rectangle(*img_rgb,
		                        cv::Point2f((*it)->px[0]-2, (*it)->px[1]-2),
		                        cv::Point2f((*it)->px[0]+2, (*it)->px[1]+2),
		                        cv::Scalar(0,255,0), CV_FILLED);
		      }


			frame_output->AddPoint(*latest_output,
					new slambench::values::FrameValue(inputSize.x, inputSize.y,
							slambench::io::pixelformat::EPixelFormat::RGB_III_888 , img_rgb->data));
		}



		if(pointcloud_output->IsActive()) {
			slambench::values::PointCloudValue *point_cloud = new slambench::values::PointCloudValue();


		    for (auto key_frame = vo_->map().keyframes_.begin();
		        key_frame != vo_->map().keyframes_.end(); ++key_frame)
		        for (auto it = (*key_frame)->fts_.begin(); it != (*key_frame)->fts_.end(); ++it) {
		            if((*it)->point == NULL)
		                continue;
		            Eigen::Vector3d pos = T_world_from_vision_ * (*it)->point->pos_;
		            slambench::values::Point3DF new_vertex;
		            new_vertex.X = (float)pos[0];
		            new_vertex.Y = (float)pos[1];
		            new_vertex.Z = (float)pos[2];
			    	point_cloud->AddPoint(new_vertex);
		        }


			// Take lock only after generating the map
			std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
			pointcloud_output->AddPoint(ts, point_cloud);

		}




	return true;
}



