cmake_minimum_required(VERSION 3.16)

project(SurroundView LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

enable_language(CUDA)

#set(CMAKE_CUDA_COMPILER "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin\\nvcc.exe")
#set(
#         CMAKE_NVCC_FLAGS
#         ${CMAKE_NVCC_FLAGS};
#         -Xcompiler
#         #-fopenmp
#         #-O3
#         -use_fast_math
#         -gencode arch=compute_62,code=sm_62
#		 -ccbin "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Enterprise\\VC\\Tools\\MSVC\\14.16.27023\\bin\\Hostx86\\x64\\cl.exe"
#)

set(CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_ARCHITECTURES 62)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)


#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})


set(PROJ_INCLUDE_DIRS)
set(PROJ_LIBRARIES)

#set(Qt5Modules Core Widgets Gui)
#find_package(Qt5 COMPONENTS ${Qt5Modules} REQUIRED)

#set(CMAKE_AUTOMOC ON)


set(3DP_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3dparty/")
list(APPEND PROJ_INCLUDE_DIRS ${3DP_INCLUDE_DIR})

list(APPEND PROJ_LIBRARIES "opengl32.lib")

list(APPEND PROJ_LIBRARIES "${3DP_INCLUDE_DIR}/glew-2.1.0/lib/Release/x64/glew32.lib")
list(APPEND PROJ_INCLUDE_DIRS "${3DP_INCLUDE_DIR}/glew-2.1.0/include")

list(APPEND PROJ_LIBRARIES "${3DP_INCLUDE_DIR}/glfw-3.3.6.bin.WIN64/lib-vc2017/glfw3.lib")
list(APPEND PROJ_INCLUDE_DIRS "${3DP_INCLUDE_DIR}/glfw-3.3.6.bin.WIN64/include")

#list(APPEND PROJ_LIBRARIES "${3DP_INCLUDE_DIR}/glm/build/glm/Release/glm_static.lib")
list(APPEND PROJ_INCLUDE_DIRS "${3DP_INCLUDE_DIR}/glm")

list(APPEND PROJ_LIBRARIES "${3DP_INCLUDE_DIR}/assimp-5.2.3/build/install/lib/assimp-vc141-mt.lib")
list(APPEND PROJ_INCLUDE_DIRS "${3DP_INCLUDE_DIR}/assimp-5.2.3/build/install/include")

file(GLOB files "${3DP_INCLUDE_DIR}/ffmpeg-5.0-full_build-shared/lib/*.lib")
foreach(file ${files})
  list(APPEND PROJ_LIBRARIES "${file}")
endforeach()
list(APPEND PROJ_INCLUDE_DIRS "${3DP_INCLUDE_DIR}/ffmpeg-5.0-full_build-shared/include")

#file(GLOB files "${PROJECT_SOURCE_DIR}/build/packages/WindowsStoreOpenGL.0.777.79/build/native/Release_x64/*.lib")
#foreach(file ${files})
  #add_test(validate_${i}, "validator", ${file})
#  list(APPEND PROJ_LIBRARIES "${file}")
#endforeach()
#list(APPEND PROJ_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/build/packages/WindowsStoreOpenGL.0.777.79/build/native/include")

#set(GLFW_LIB "${3DP_INCLUDE_DIR}/glfw-3.3.6.bin.WIN64/lib-vc2017/glfw3.lib")
#set(GLES_LIB "${3DP_INCLUDE_DIR}/mesa3d-22.0.0.2-rc2/lib/x64/src/egl/libEGL.lib")
#list(APPEND GLES_LIB "${3DP_INCLUDE_DIR}/lib/libGLESv2.so")

#set(GLM_INCLUDE_DIR "${3DP_INCLUDE_DIR}/glm")

find_package(OpenCV REQUIRED)
list(APPEND PROJ_LIBRARIES ${OpenCV_LIBS})
list(APPEND PROJ_LIBRARIES ${CUDA_LIBRARIES})
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND PROJ_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

#find_package(OpenGL REQUIRED)
#list(APPEND PROJ_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIR})

#set(OPENGL_EGL_INCLUDE_DIR "${3DP_INCLUDE_DIR}/mesa-22.0.0-rc2/include")
#list(APPEND PROJ_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIR})

#list(APPEND PROJ_LIBRARIES ${GLES_LIB})


include_directories(${PROJECT_SOURCE_DIR}/include)


file(GLOB SOURCE_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cusrc/*.cu
)


add_executable(SurroundView ${SOURCE_FILES})

set_target_properties(SurroundView 
			PROPERTIES OUTPUT_NAME "SurroundView" 
			CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET SurroundView PROPERTY CUDA_STANDARD 14)

target_link_libraries(SurroundView PUBLIC ${PROJ_LIBRARIES})
target_include_directories(SurroundView PUBLIC ${PROJ_INCLUDE_DIRS})

target_compile_options(SurroundView PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)
