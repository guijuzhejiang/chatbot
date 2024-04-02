#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "math.h"

namespace py = pybind11;

/**
* @brief 浮点数转字符串
* @param float_var    参数1  浮点数
*
* @return (string)返回结果
*/
std::string float2str(float float_var){
    std::ostringstream oss;
    oss<<float_var;
    std::string str(oss.str());

    return str;
}

/**
* @brief 判断文件是否存在
* @param name    参数1  文件路径
*
* @return (bool)返回结果
*/
bool isFileExists_ifstream(std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

/**
* @brief 计算余弦相似度
* @param _embed_fp    参数1  存放特征的文件路径
* @param _idx_fp      参数2  存放索引的文件路径
*
* @return (double)返回结果
*/
double calculSimilar(py::array_t<float> & v1, py::array_t<float> & v2)
{
    py::buffer_info buf1 = v1.request();
    py::buffer_info buf2 = v2.request();

    if (buf1.ndim !=1 || buf2.ndim !=1)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }

    if (buf1.size !=buf2.size)
    {
        throw std::runtime_error("Input shape must match");
    }

    //获取numpy.ndarray 数据指针
    float* ptr1 = (float*)buf1.ptr;
    float* ptr2 = (float*)buf2.ptr;

    double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
    for (int i = 0; i < buf1.shape[0]; i++)
    {
        ret += ptr1[i] * ptr2[i];
        mod1 += ptr1[i] * ptr1[i];
        mod2 += ptr2[i] * ptr2[i];
    }
    return ret / sqrt(mod1) / sqrt(mod2) ;
}

class FeatureDB {
private:
    void validate_db() {
        bool embed_file_is_exist = isFileExists_ifstream(this->embed_fp);
        bool index_file_is_exist = isFileExists_ifstream(this->idx_fp);

        bool invalid = false;

        if ((!embed_file_is_exist && index_file_is_exist) || (embed_file_is_exist && !index_file_is_exist)) {
            invalid = true;
        } else {
            // read feature
            std::ifstream _embed_file;
            _embed_file.open(this->embed_fp, std::ios::in | std::ios::binary);
            int _feature_cnt = 0;
            auto result = py::array_t<float>(this->feature_dims);
            py::buffer_info buf = result.request();
            float *ptr = (float *) buf.ptr;
            while (_embed_file.read(reinterpret_cast<char *>(&ptr[0]),
                                    this->feature_dims * sizeof(ptr[0]))) {
                _feature_cnt++;
            }

            _embed_file.close();

            // read index
            std::ifstream _idx_file;
            _idx_file.open(this->idx_fp, std::ios::in);

            int _index_cnt = 0;
            std::string _strLine = "";
            while (std::getline(_idx_file, _strLine)) {
                _index_cnt++;
            }

            _idx_file.close();

            if(_index_cnt != _feature_cnt){
                invalid = true;
            }
        }

        if (invalid) {
            if (embed_file_is_exist) std::remove(this->embed_fp.c_str());
            if (index_file_is_exist) std::remove(this->idx_fp.c_str());
        }
    };


public:
    float threshold = 0.298;
    int feature_dims = 192; // 特征维度
    std::string idx_fp; // index文件路径
    std::string embed_fp; // 特征文件路径
    std::vector <py::array_t<float>> features; // 特征buffer
    std::vector <std::string> idxs; // index buffer

    /**
    * @brief 构造函数，定义存放index和特征的文件路径
    * @param _embed_fp    参数1  存放特征的文件路径
    * @param _idx_fp      参数2  存放索引的文件路径
    */
    FeatureDB(const std::string &_embed_fp, const std::string &_idx_fp, const float &_threshold, const int &_feature_dims) : embed_fp(_embed_fp), idx_fp(_idx_fp), threshold(_threshold), feature_dims(_feature_dims) {
        std::cout << "init db:" << this->embed_fp << std::endl;
        this->validate_db();

        // read feature
        std::ifstream _embed_file;
        _embed_file.open(this->embed_fp, std::ios::in | std::ios::binary);
        
        int _loop_cnt = 0;
        auto result = py::array_t<float>(this->feature_dims);
        py::buffer_info buf = result.request();
        float *ptr = (float *) buf.ptr;
        while (_embed_file.read(reinterpret_cast<char *>(&ptr[0]),
                                this->feature_dims * sizeof(ptr[0]))) {
            _loop_cnt++;
            features.push_back(result);
            result = py::array_t<float>(this->feature_dims);
            buf = result.request();
            ptr = (float *) buf.ptr;
        }

        _embed_file.close();

        // read index
        std::ifstream _idx_file;
        _idx_file.open(_idx_fp, std::ios::in);

        std::string _strLine = "";
        int index = 0;
        while (std::getline(_idx_file, _strLine)) {
            this->idxs.push_back(_strLine);
            index++;
        }

        _idx_file.close();
        
        std::cout << "init ok." << std::endl;
    }

    /**
    * @brief 查询特征数据库
    * @param _id    参数1 - 特征索引
    *
    * @return 返回说明
    *     true  - 目标特征
    *     false - 空ndarray
    */
    py::array_t<float> search_by_index(const std::string &_id) {
        std::cout << "search:" << _id << std::endl;
        // find feature index
        int target_index = -1;
        int idxs_size = this->idxs.size();
        for (int i = 0; i < idxs_size; i++) {
            if (this->idxs[i].compare(_id) == 0) {
                target_index = i;
                break;
            }
        }

        if (target_index == -1) {
            return py::array_t<float>(this->feature_dims);
        } else {
            return this->features[target_index];
        }
    }

    /**
    * @brief 查询特征数据库
    * @param _input_feature ndarray特征
    *
    * @return 特征索引
    */
    std::string search(py::array_t<float> &_input_feature) {
        std::cout << "searching..." << std::endl;
        if (!isFileExists_ifstream(this->idx_fp) || !isFileExists_ifstream(this->embed_fp) || this->idxs.size()==0) {
            return "null|0.0";
        }

        // find feature index
        float similarity = 0.0;
        float _max_sim_value = -1;

        int _max_sim_idx = -1;
        for (int i = 0; i < this->idxs.size(); i++) {
            similarity = calculSimilar(_input_feature, this->features[i]);
//            std::cout << similarity << std::endl;
            if (similarity > _max_sim_value) {
                _max_sim_value = similarity;
                _max_sim_idx = i;
            }
        }


        std::string res = "";
        if (_max_sim_value < this->threshold) {
            std::stringstream fmt;
            // 造字符串流
            fmt << "null@" << this->idxs[_max_sim_idx] << "|" << float2str(_max_sim_value);
            res = fmt.str();
        } else {
            res.append(this->idxs[_max_sim_idx]);
            res.append("|");
            res.append(float2str(_max_sim_value));
        }
//        std::cout << _max_sim_value << std::endl;
        return res;
    }

    /**
    * @brief 插入特征数据库
    * @param _input_feature ndarray特征
    * @param _id            索引名
    *
    * @return (std::string）返回结果字符串
    */
    std::string add(py::array_t<float> &_input_feature, const std::string &_id) {
        std::cout << "add:" << _id << std::endl;
        // get feature ndarray info
        py::buffer_info buf = _input_feature.request();
        // get pointer
        float *ptr = (float *) buf.ptr;

        // append features
        std::ofstream _output_embed_file;
        _output_embed_file.open(this->embed_fp,
                                std::ios::out | std::ios::app | std::ios::binary);

        if (buf.size > 0) {
            _output_embed_file.write(reinterpret_cast<char *>(&ptr[0]),
                                     _input_feature.shape()[0] * sizeof(ptr[0]));
        }
        _output_embed_file.close();

        this->features.push_back(_input_feature);

        // append index
        std::ofstream _output_idx_file;
        _output_idx_file.open(this->idx_fp, std::ios::out | std::ios::app);
        _output_idx_file << _id << "\n";
        _output_idx_file.close();

        this->idxs.push_back(_id);

        return "success";
    }

};


PYBIND11_MODULE(feature, m) {
py::class_<FeatureDB>(m, "FeatureDB")
.def(py::init<const std::string &, const std::string &, const float &, const int &>())
.def("add", &FeatureDB::add)
.def("search_by_index", &FeatureDB::search_by_index)
.def("search", &FeatureDB::search);
}