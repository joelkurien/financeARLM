#ifndef TENSOR_H
#define TENSOR_H

#include<iostream>
#include<Eigen/Dense>
#include<vector>
#include<limits>

using namespace std;
using namespace Eigen;

class Tensor {
    private:
        MatrixXd matrix;
        size_t mrows;
        size_t mcols;

        class PositionProxy{
            private:
                Tensor& tensor;
                size_t row;
                size_t col;
            public:
                PositionProxy(Tensor& t,size_t r): tensor(t),row(r),col(-1) {}
                size_t operator[] (size_t c){
                    col = c;
                    return tensor.at(row, col);
                }
        };

    public:
        Tensor(size_t rows, size_t cols)
            :mrows(rows), mcols(cols) {
            MatrixXd b(rows, cols);
            matrix = b;
        }

        PositionProxy operator[](size_t row){
            return PositionProxy(*this, row);
        }

        tuple<size_t, size_t> shape(){
            return {matrix.rows(), matrix.cols()};
        }
        
        double at(size_t row, size_t col){
            if(row < 0){
                if(mrows+row < 0) throw "Invalid row index";
                row = row < 0 ? mrows+row : row;
            }
            if(col < 0){
                if(mcols+col<0) throw "Invalid col index";
                col = col < 0 ? mcols+col : col;
            }
            if(row < mrows && col < mcols) return matrix(row, col);
            else out_of_range("Row or Col index out of bounds");
            return -numeric_limits<double>::infinity();
        }

        void put(size_t row, size_t col, double val) {
            if(row<0 || row > mrows || col<0 || col > mcols) out_of_range("Index out of bounds");
            matrix(row, col) = val;
        }

        size_t size(){
            auto [row, col] = shape();
            return row*col;
        }

        bool empty(){
            auto [row, col] = shape();
            return (row == 0 && col == 0);
        }
        
        
};
#endif