/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam {

MapPoint::MapPoint(long id, Vec3 position) : id_(id), pos_(position) {}

MapPoint::Ptr MapPoint::CreateNewMappoint() {
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}


// 1.处理旧活跃关键帧时将其左右图像每个特征点关联的地图（路标）点清除，并将路标点对应的此特征点删去，以后用不到了所以删掉，对应后面有些3d点用不到了删去CleanMap()  2.后端处理异常点
void MapPoint::RemoveObservation(std::shared_ptr<Feature> feat) {
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto iter = observations_.begin(); iter != observations_.end();
         iter++) {
        if (iter->lock() == feat) {//weak_ptr.lock()获取所管理对象的强引用，即可将weak_ptr赋给shared_ptr  ， https://www.cnblogs.com/osbreak/p/9209104.html
            observations_.erase(iter);
            feat->map_point_.reset();
            observed_times_--;
            break;
        }
    }
}

}  // namespace myslam
