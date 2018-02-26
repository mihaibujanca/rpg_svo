cmake_minimum_required(VERSION 2.8)
MESSAGE(STATUS "Entering SVO CMakeLists.txt")
project(svo)

# Build Sophus dep
ADD_SUBDIRECTORY(svo/thirdparty/Sophus)
ADD_SUBDIRECTORY(svo/thirdparty/vikit_common)

################################################################################
# user build settings

MESSAGE(STATUS "Prior to setting HAVE_G2O I found HAVE_G2O = ${HAVE_G2O}" )
SET(TRACE FALSE)
SET(HAVE_G2O TRUE)
SET(DEBUG_OUTPUT FALSE) # Only relevant if build without ROS
SET(INFO_OUTPUT FALSE) # Disable output for SLAMbench
MESSAGE(STATUS "Now HAVE_G2O = ${HAVE_G2O}" )

################################################################################

SET(PROJECT_NAME svo)
PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8.3)
SET(CMAKE_BUILD_TYPE Release) # Release, RelWithDebInfo
SET(CMAKE_VERBOSE_MAKEFILE OFF)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

# Set definitions
IF(TRACE)
  ADD_DEFINITIONS(-DSVO_TRACE)
ENDIF()
IF(HAVE_G2O)
  ADD_DEFINITIONS(-DUSE_BUNDLE_ADJUSTMENT)
ENDIF()
IF(DEBUG_OUTPUT)
  ADD_DEFINITIONS(-DSVO_DEBUG_OUTPUT)
ENDIF()
IF(INFO_OUTPUT)
  ADD_DEFINITIONS(-DSVO_INFO_OUTPUT)
ENDIF()

# Set build flags, set ARM_ARCHITECTURE environment variable on Odroid
# Set build flags. Set IS_ARM on odroid board as environment variable
SET(CMAKE_CXX_FLAGS "-Wall -D_LINUX -D_REENTRANT -march=native -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas  -Wno-misleading-indentation  -Wno-int-in-bool-context")
IF(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon -march=armv7-a")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx -msse -msse -msse2 -msse3 -mssse3")
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ENDIF()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Add plain cmake packages
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system)

# Include dirs
INCLUDE_DIRECTORIES(
  svo/include
  ${EIGEN3_INCLUDE_DIR}
  ${Sophus_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${vikit_common_SOURCE_DIR}/include
)

# Set link libraries
LIST(APPEND LINK_LIBS
  ${OpenCV_LIBS}
  Sophus
  vikit_common
  ${Boost_LIBRARIES}
)

# Set sourcefiles
LIST(APPEND SOURCEFILES
  svo/src/frame_handler_mono.cpp
  svo/src/frame_handler_base.cpp
  svo/src/frame.cpp
  svo/src/point.cpp
  svo/src/map.cpp
  svo/src/pose_optimizer.cpp
  svo/src/initialization.cpp
  svo/src/matcher.cpp
  svo/src/reprojector.cpp
  svo/src/feature_alignment.cpp
  svo/src/feature_detection.cpp
  svo/src/depth_filter.cpp
  svo/src/config.cpp
  svo/src/sparse_img_align.cpp)

# Add g2o if available
IF(HAVE_G2O)
	MESSAGE(STATUS "SVO is Using G2O.")
 	 FIND_PACKAGE(G2O REQUIRED)
  	FIND_PACKAGE(SuiteSparse REQUIRED)
 	 INCLUDE_DIRECTORIES(
    ${G2O_INCLUDE_DIR}
    ${CHOLMOD_INCLUDE_DIR}
  	)
 	 LIST(APPEND LINK_LIBS
 	   ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE}
 	   ${G2O_SOLVER_PCG} ${G2O_TYPES_SBA} ${CHOLMOD_LIBRARIES}
 	 )
	  LIST(APPEND SOURCEFILES svo/src/bundle_adjustment.cpp)
ENDIF(HAVE_G2O)

# Create svo library
ADD_LIBRARY(svo SHARED ${SOURCEFILES})
TARGET_LINK_LIBRARIES(svo ${LINK_LIBS})

add_library(svo-original SHARED
 svo_library.cpp
)

target_link_libraries(svo-original
  svo
)

generate_slam_library(svo-original "")
