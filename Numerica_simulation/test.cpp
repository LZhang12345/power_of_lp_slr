#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <chrono>
#include <map>
#include <typeinfo>
using namespace Eigen;
using namespace std;
//namespace fs = std::__fs::filesystem;

MatrixXd find_x(const VectorXd &h, const VectorXd &r, const VectorXd &v, double mu) {
    int m = h.size();
    int n = r.size();
    MatrixXd X = MatrixXd::Zero(m, n);
    VectorXd deri = v + mu * r;

    vector<pair<double, int>> pairs;
    for (int i = 0; i < n; ++i) {
        pairs.push_back(make_pair(deri(i), i));
    }

    sort(pairs.begin(), pairs.end(), greater<pair<double, int>>());

    for (int i = 0; i < m; ++i) {
        X(i, pairs[i].second) = 1;
    }

    return X;
}

vector<double> search_mu(const VectorXd &h, const VectorXd &r, const VectorXd &v, double lamb, double a, int ite, double eps) {
    double mu_high = 1.0;
    double mu_low = 0.0;
    vector<double> results; 
    auto start = std::chrono::high_resolution_clock::now();
    int i = 1;
    while ((h.transpose() * find_x(h, r, v, mu_high) * r) < lamb * a && i < ite) {
        mu_low = mu_high;
        mu_high *= 2;
        i++;
    }

    i = 1;
    double mu_temp = 0.5 * (mu_high + mu_low);
    while (i < ite && abs(mu_high - mu_low) > eps) {
        if ((h.transpose() * find_x(h, r, v, mu_temp) * r) >= lamb * a) {
            mu_high = mu_temp;
        } else {
            mu_low = mu_temp;
        }
        mu_temp = 0.5 * (mu_high + mu_low);
        i++;
    }
    auto stop = std::chrono::high_resolution_clock::now();
    chrono::duration<double> duration = stop - start;
    results.push_back(duration.count());
    results.push_back(h.transpose() * find_x(h, r, v, mu_high) * v); //opt mu high
    results.push_back(h.transpose() * find_x(h, r, v, mu_low) * v); // opt mu low
    results.push_back(h.transpose() * find_x(h, r, v, mu_high) * r); // rel mu high
    results.push_back(h.transpose() * find_x(h, r, v, mu_low) * r); // rel mu low
    return results;
}

VectorXd readVectorFromFile(const string& filename) {
    vector<double> values;
    double val;
    ifstream file(filename);

    while (file >> val) {
        values.push_back(val);
    }
    // cout << "Read " << values.size() << " values from " << filename << endl;
    return Map<VectorXd>(values.data(), values.size());
}

int main(int argc, char** argv) {
    //double lamb = 0.95;
    vector<double> inputVector = {.1, .2, .3, .4, .5, .6, .7, .8, .9, .925, .95, .975};
    //vector<double> resultVector;
    vector<double> mList = {10, 20, 50, 100, 200, 500};
    vector<double> nList = {50, 100, 200, 500, 1000, 2000};
    //int m = 50;
    //int n = 500;
    string start_num = argv[1];
    double eps = 1e-4;
    double lamb = .95;
    int ite = 500;
    string current_path = "/project/large_scale_lp/ComputationalResult/NewAlgo1/Solve/";
    string parent_path = "/project/large_scale_lp/ComputationalResult/NewAlgo1/";//current_path.substr(0, current_path.length());

    string basePath = parent_path + "Instances_verify/";
    int start_num_int = stoi(start_num);
    vector<int> arg_numbers;
    for (int i = start_num_int; i <= 9999; i += 500) {
        arg_numbers.push_back(i);
    } 
    
    for(int arg_int : arg_numbers){
    map<string, vector<double>> resultMap;
    string arg = to_string(arg_int);
    for(int m : mList) {
    for(int n : nList) {
    if(m > n){continue;}
//    cout << "value of lamb " << lamb << endl;
    string dataPath = to_string(static_cast<int>(lamb * 1000)) + "_" + to_string(m) + "_" + to_string(n);
    string instancePath = basePath + dataPath + "/data/" + dataPath;
    string hPath = instancePath + "_" + arg + "_h_data.txt";
    string vPath = instancePath + "_" + arg + "_v_data.txt";
    string rPath = instancePath + "_" + arg + "_r_data.txt";
//    cout << "process 1 " << endl;
    VectorXd h_temp = readVectorFromFile(hPath);
    VectorXd v = readVectorFromFile(vPath);
    VectorXd r = readVectorFromFile(rPath);
//    cout << "process 2 " << h.size() << " " << v.size() << " " << r.size()  << endl;
    // Compute 'a'
//    cout << "process 3 " << endl;
    VectorXd h = h_temp;
    std::sort(h.data(), h.data() + h.size(), std::greater<double>());
    VectorXd sorted_r = r;
    std::sort(sorted_r.data(), sorted_r.data() + sorted_r.size(), std::greater<double>());
//    cout << "process 4 " << endl;
    double a = h.head(m).dot(sorted_r.head(m));
    
    // Start measuring time
    auto start = std::chrono::high_resolution_clock::now();

    // Compute 'rel'
//    cout << "process 5 " << endl;
    double rel = (h.transpose() * find_x(h, r, v, 0) * r);
//    cout << "Relevance: " << rel << ", Lambda * a: " << lamb * a << endl;
//    cout << "process 6 " << endl;
    vector<double> tempVec;;
    if (rel >= lamb * a) {
//        cout << "process 7 " << endl;
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = stop - start;
        tempVec.push_back(duration.count());
        tempVec.push_back(h.transpose() * find_x(h, r, v, 0) * v); // opt 
        tempVec.push_back(h.transpose() * find_x(h, r, v, 0) * v); // opt
        tempVec.push_back(h.transpose() * find_x(h, r, v, 0) * r); // rel
        tempVec.push_back(h.transpose() * find_x(h, r, v, 0) * r); // rel
    } else {
        tempVec = search_mu(h, r, v, lamb, a, ite, eps);
    }

    // Stop measuring time and calculate the elapsed time
    // auto stop = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = stop - start;

    resultMap[arg + "_" + to_string(m) + "_" + to_string(n)] = tempVec; 
    //delete h,v,r,sorted_r,a,rel,opt;
    }
    }

    // Save the results to a file
    ofstream outputFile("/project/large_scale_lp/ComputationalResult/NewAlgo1/Solve/Results_cpp_verify/" + arg + ".txt");
    if (outputFile.is_open()) {
        for (const auto& pair : resultMap) {
            outputFile << pair.first << " : ";
            for (const double& value : pair.second) {
                outputFile << value << ",";
            }
            outputFile << std::endl;
        }
        outputFile.close();
        //std::cout << "Map saved to myMap.txt" << std::endl;
    } else {
        std::cerr << "Failed to open file for writing." << std::endl;
    }
    }
    return 0;
}

