# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xuwenyu/learn_slam/ch9

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xuwenyu/learn_slam/ch9/build

# Include any dependencies generated for this target.
include CMakeFiles/bundle_adjustment_ceres.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bundle_adjustment_ceres.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bundle_adjustment_ceres.dir/flags.make

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o: CMakeFiles/bundle_adjustment_ceres.dir/flags.make
CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o: ../bundle_adjustment_ceres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xuwenyu/learn_slam/ch9/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o -c /home/xuwenyu/learn_slam/ch9/bundle_adjustment_ceres.cpp

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xuwenyu/learn_slam/ch9/bundle_adjustment_ceres.cpp > CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.i

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xuwenyu/learn_slam/ch9/bundle_adjustment_ceres.cpp -o CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.s

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.requires:

.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.requires

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.provides: CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.requires
	$(MAKE) -f CMakeFiles/bundle_adjustment_ceres.dir/build.make CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.provides.build
.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.provides

CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.provides.build: CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o


# Object files for target bundle_adjustment_ceres
bundle_adjustment_ceres_OBJECTS = \
"CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o"

# External object files for target bundle_adjustment_ceres
bundle_adjustment_ceres_EXTERNAL_OBJECTS =

bundle_adjustment_ceres: CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o
bundle_adjustment_ceres: CMakeFiles/bundle_adjustment_ceres.dir/build.make
bundle_adjustment_ceres: /usr/local/lib/libceres.a
bundle_adjustment_ceres: libbal_common.a
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libglog.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libspqr.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libtbb.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libcholmod.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libccolamd.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libcamd.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libcolamd.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libamd.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/liblapack.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libf77blas.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libatlas.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/librt.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/liblapack.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libf77blas.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libatlas.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/librt.so
bundle_adjustment_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bundle_adjustment_ceres: CMakeFiles/bundle_adjustment_ceres.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xuwenyu/learn_slam/ch9/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bundle_adjustment_ceres"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bundle_adjustment_ceres.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bundle_adjustment_ceres.dir/build: bundle_adjustment_ceres

.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/build

CMakeFiles/bundle_adjustment_ceres.dir/requires: CMakeFiles/bundle_adjustment_ceres.dir/bundle_adjustment_ceres.cpp.o.requires

.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/requires

CMakeFiles/bundle_adjustment_ceres.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bundle_adjustment_ceres.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/clean

CMakeFiles/bundle_adjustment_ceres.dir/depend:
	cd /home/xuwenyu/learn_slam/ch9/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xuwenyu/learn_slam/ch9 /home/xuwenyu/learn_slam/ch9 /home/xuwenyu/learn_slam/ch9/build /home/xuwenyu/learn_slam/ch9/build /home/xuwenyu/learn_slam/ch9/build/CMakeFiles/bundle_adjustment_ceres.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bundle_adjustment_ceres.dir/depend

