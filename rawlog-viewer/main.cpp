#include <mrpt/vision/CVideoFileWriter.h>
#include <mrpt/containers/stl_containers_utils.h>
#include <mrpt/system/CTicTac.h>
#include <mrpt/io/CFileGZInputStream.h>
#include <mrpt/io/CFileGZOutputStream.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/memory.h>
#include <mrpt/system/CDirectoryExplorer.h>
#include <mrpt/poses/CPosePDFParticles.h>
#include <mrpt/poses/CPosePDFGaussian.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/maps/COccupancyGridMap2D.h>
#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/slam/CICP.h>
#include <mrpt/system/datetime.h>
#include <mrpt/math/ops_matrices.h>  // << ops
#include <mrpt/math/ops_vectors.h>  // << ops
#include <mrpt/math/wrap2pi.h>
#include <mrpt/core/aligned_std_map.h>

#include <mrpt/obs/CObservationComment.h>
#include <mrpt/obs/CObservationOdometry.h>
#include <mrpt/obs/CObservation2DRangeScan.h>
#include <mrpt/obs/CObservationBeaconRanges.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <mrpt/obs/CObservation3DRangeScan.h>
#include <mrpt/obs/CObservationGasSensors.h>
#include <mrpt/obs/CObservationBearingRange.h>
#include <mrpt/obs/CObservationRange.h>

#include <mrpt/serialization/CArchive.h>

#define MRPT_NO_WARN_BIG_HDR
#include <mrpt/obs.h>

#include <mrpt/maps/CSimplePointsMap.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/poses/CPosePDFParticles.h>

#include <cstdlib>
#include <functional>
#include <thread>
#include <string>
#include <mutex>
#include <memory>
#include <iostream>
#include <iomanip>
#include "CRawlogTreeProcessor.h"
using namespace mrpt;
using namespace mrpt::opengl;
using namespace mrpt::maps;
using namespace mrpt::img;
using namespace mrpt::math;
using namespace mrpt::obs;
using namespace mrpt::system;
using namespace mrpt::serialization;
using namespace mrpt::poses;
using namespace mrpt::rtti;
using namespace mrpt::config;
using namespace mrpt::vision;
using namespace mrpt::io;
using namespace std;

int main(int argc, char* argv[]) {
	CFileGZInputStream fil(argv[1]);
	uint64_t fil_size = fil.getTotalBytesCount();
  CRawlog rawlog;
  rawlog.clear();
  size_t count_loop = 0;
  int entry_index = 0;
  bool keep_loading = true;
  bool already_warned_too_large_file = false;
  string err_msg;
  int first = 0;
  int last = -1;
  while(keep_loading)
  {
    if(count_loop++ % 10 == 0)
    {
      uint64_t fil_pos = fil.getPosition();
      static double last_ratio = -1;
      double ratio = fil_pos / (1.0 * fil_size);

      if(ratio - last_ratio >= 0.006)
      {
        last_ratio = ratio;

        unsigned long memUsg = getMemoryUsage();
        double memUsg_Mb = memUsg / (1024.0 * 1024.0);
        if (memUsg_Mb > 2600 && !already_warned_too_large_file)
        {
          already_warned_too_large_file = true;
          err_msg += string("Memory Usage exceeded 2600 MB.");
          keep_loading = false;
        }
      }
    }
    // Try to load the Object into a serializable Object
    CSerializable::Ptr new_obj;
    try
    {
      archiveFrom(fil) >> new_obj;
      //Check type:
      if (new_obj->GetRuntimeClass() == CLASS_ID(CSensoryFrame))
      {
        if (entry_index >= first && (last == -1 || entry_index <= last))
          rawlog.addObservationsMemoryReference(
            std::dynamic_pointer_cast<CSensoryFrame>(new_obj)
            );
        entry_index++;
      }
      else if (new_obj->GetRuntimeClass() == CLASS_ID(CSensoryFrame))
      {
        if (entry_index >= first && (last == -1 || entry_index <= last))
          rawlog.addActionsMemoryReference(
            std::dynamic_pointer_cast<CActionCollection>(new_obj)
            );
            entry_index++;
      }
      /* Added in MRPT 0.6.0: The new "observations only" format: */
      else if (new_obj->GetRuntimeClass()->derivedFrom(
            CLASS_ID(CObservation)))
      {
        if (entry_index >= first && (last == -1 || entry_index <= last))
          rawlog.addObservationMemoryReference(
            std::dynamic_pointer_cast<CObservation>(new_obj));
        entry_index++;
      }
      /* FOR BACKWARD COMPATIBILITY: CPose2D was used previously instead
        of an "ActionCollection" object
                                        26-JAN-2006	*/
      else if (new_obj->GetRuntimeClass() == CLASS_ID(CPose2D))
      {
        if (entry_index >= first && (last == -1 || entry_index <= last))
        {
          CPose2D::Ptr poseChange =
            std::dynamic_pointer_cast<CPose2D>(new_obj);
          CActionCollection::Ptr temp =
            mrpt::make_aligned_shared<CActionCollection>();
          CActionRobotMovement2D action;
          CActionRobotMovement2D::TMotionModelOptions options;
          action.computeFromOdometry(*poseChange, options);
          temp->insert(action);

          rawlog.addActionsMemoryReference(temp);
        }
        entry_index++;
      }
      else if (new_obj->GetRuntimeClass() == CLASS_ID(CRawlog))
      {
        CRawlog::Ptr rw = std::dynamic_pointer_cast<CRawlog>(new_obj);
        rawlog = std::move(*rw);
      }
      else
      {
        // Unknown class:
        // New in MRPT v1.5.0: Allow loading some other classes:
        rawlog.addGenericObject(new_obj);
      }

      // Passed last?
      if (last != -1 && entry_index > last) keep_loading = false;
    }
    catch (std::bad_alloc&)
    {
      // Probably we're in a 32 bit machine and we rose up to 2Gb of
      // mem... free
      //  some, give a warning and go on.
      if (rawlog.size() > 10000)
      {
        size_t NN = rawlog.size() - 10000;
        while (rawlog.size() > NN) rawlog.remove(NN);
      }
      else
        rawlog.clear();

      string s =
        "OUT OF MEMORY: The last part of the rawlog has been freed "
          "to allow the program to continue.\n";
#if MRPT_WORD_SIZE == 32
      s +=
        "  This is a 32bit machine, so the maximum memory available is "
        "2Gb despite of the real RAM installed.";
#endif
      err_msg += s;
      keep_loading = false;
    }
    catch (exception& e)
    {
      err_msg += e.what();
      keep_loading = false;
    }
    catch (...)
    {
      keep_loading = false;
    }
  } // end while keep loading
    std::shared_ptr<CRawlogTreeProcessor> m_tree = std::make_shared<CRawlogTreeProcessor>();
    m_tree->setRawlogSource(&rawlog);

  // Call different parameters
  for(size_t i = 0;i < rawlog.size(); i++){
    std::cout << i << std::endl;
    m_tree->getTreeDataPoint(i);
  }

  return 0;
}
