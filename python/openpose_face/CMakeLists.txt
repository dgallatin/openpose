set(PYTHON_FILES
    openpose_face.py
    __init__.py
    _openposeFace.cpp)

add_library(_openposeface SHARED ${PYTHON_FILES})
target_link_libraries(_openposeface openpose ${OpenPose_3rdparty_libraries})
SET_TARGET_PROPERTIES(_openposeface PROPERTIES PREFIX "")
configure_file(openpose_face.py openpose_face.py)
configure_file(__init__.py __init__.py)

#install(TARGETS _openposeface DESTINATION python)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose_face FILES_MATCHING PATTERN "*.so")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ DESTINATION python/openpose_face FILES_MATCHING PATTERN "*.py")

