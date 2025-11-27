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
        Matrix<double, Dynamic, Dynamic, RowMajor> matrix;
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
        
        tuple<double*, size_t, size_t, size_t> view(int rstart, int rend, int cstart, int cend){
            double* basePtr = matrix.data();
            double* subBasePtr = basePtr + (mcols*rstart + cstart);
            double height = rend-rstart+1;
            double width = cend-cstart+1;
            return {subBasePtr, height, width, mcols};
        }

        Tensor slice(int rstart, int cstart, int rend, int cend){
            auto[smaddr, height, width, mcols] = view(rstart, rend, cstart, cend);
            if(rstart < 0 || rstart > mrows || rend < 0 || rend > mrows )
                throw out_of_range("Row Index out of bounds");
            if(cstart < 0 || cstart > mcols || cend < 0 || cend > mcols)
                throw out_of_range("Col Index out of bounds");
            
            Tensor slc(height, width);
            for(int i=0; i<height; i++){
                for(int j=0; j<width; j++){
                    slc.put(i,j,*(smaddr + mcols*i + j));
                }
            }
            return slc;
        }
};


class Tensor3D {
    private:
        vector<Tensor> batch;
        size_t nbatch;
        size_t mrows;
        size_t mcols;
    
    public:
        Tensor3D(size_t nb, size_t rows, size_t cols)
            : nbatch(nb), mrows(rows), mcols(cols) {
                for(int i=0; i<nb; i++){
                    Tensor t(rows, cols);
                    batch.push_back(t);
                }
            }

        Tensor operator[](size_t b){
            return batch[b];
        }

        tuple<size_t, size_t, size_t> shape(){
            return {nbatch, mrows, mcols};
        }

        double at(size_t b, size_t r, size_t c){
            if(b < 0 || b > nbatch) out_of_range("Invalid batch index");
            return batch[b].at(r,c);
        }

        void put(size_t b, size_t r, size_t c, double val){
            if(b < 0 || b > nbatch) out_of_range("Invalid batch index");
            return batch[b].put(r, c, val);
        }

        void put(Tensor t) { return batch.push_back(t);}

        size_t size(){
            return nbatch*mrows*mcols;
        }

        bool empty(){
            if(batch.size() == 0) 
                return true;
            return false;
        }

        Tensor3D sliceBatch(size_t start, size_t end){
            size_t nb = end-start;
            if(end > nbatch || start > nbatch || end < 0 || start < 0) out_of_range("Boundaries out of range");
            Tensor3D subBatch(nb, mrows, mcols); 
            for(int i=start; i<end; i++){
                subBatch.put(batch[i]);
            }
            return subBatch;
        }
};
#endif