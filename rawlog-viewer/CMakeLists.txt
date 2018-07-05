set(LIB_SRC
  CRawlogTreeProcessor.cpp
  main_display_processor.cpp
)

add_library(rawlog_app ${LIB_SRC})

add_executable(rawlog-viewer
	main.cpp
)
target_link_libraries(rawlog_app
PUBLIC
${MRPT_LIBS}
)
target_link_libraries(rawlog-viewer
PUBLIC
rawlog_app
${CMAKE_THREAD_LIBS_INIT}
${MRPT_LIBS})
