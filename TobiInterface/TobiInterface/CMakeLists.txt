add_executable(TobiInterface TobiInterface.cpp)

# Local libraries
include_directories(${CLM_SOURCE_DIR}/include)

include_directories(../../lib/local/CLM/include)
			
target_link_libraries(TobiInterface CLM)

target_link_libraries(TobiInterface dlib)

if(WIN32)
	target_link_libraries(TobiInterface ${OpenCVLibraries})
endif(WIN32)
if(UNIX)
    target_link_libraries(TobiInterface ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
endif(UNIX)

install (TARGETS TobiInterface DESTINATION ${CMAKE_BINARY_DIR}/bin)
