#ifndef NDITERATOR_H
#define NDITERATOR_H

#include<iostream>
#include<vector>

class NDiterator{
    private:
        std::vector<size_t> shape;
        std::vector<size_t> current_shape;
        bool is_end;

    public:
        NDiterator(const std::vector<size_t>& _shape, bool end = false);

        const std::vector<size_t>& operator*() const;
        NDiterator& operator++();
        bool operator !=(const NDiterator& iter);

        static NDiterator begin(const std::vector<size_t>& shape);
        static NDiterator end(const std::vector<size_t>& shape);
};

class NDRange {
    private:
        std::vector<size_t> shape;
    public:
        NDRange(std::vector<size_t>& _shape);

        NDiterator begin();
        NDiterator end();
        size_t size();
};

#endif