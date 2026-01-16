#include "tensor.h"
#include <cstring>
#include <tuple>
#include <random>
#include <cmath>


Tensor::Tensor(std::vector<size_t> shape_list)
{
    shapes = shape_list;
    make2d(shapes);
    dim = shapes.size();
    strides = computeStrides(shapes);
    valvec.resize(size(),0);
    basePtr = valvec.data();
}

Tensor::Tensor(std::vector<double> vec, std::vector<size_t> shape_list)
    : shapes(shape_list), valvec(vec)
{
    make2d(shapes);
    dim = shapes.size();
    basePtr = valvec.data();
    strides = computeStrides(shapes);
}

Tensor::Tensor(std::vector<double> vec, std::vector<size_t> shape_list, std::vector<size_t> _strides)
    : shapes(shape_list), valvec(vec), strides(_strides)
{
    make2d(shapes);
    dim = shapes.size();
    basePtr = valvec.data();
}

Tensor::Tensor(double* ptr, std::vector<size_t> shape_list, std::vector<size_t> strides)
    : basePtr(ptr), strides(strides), shapes(shape_list)
{
    make2d(shapes);
    dim = shapes.size();
}

Tensor::Tensor(double* ptr, std::vector<size_t> shape_list)
    : basePtr(ptr), shapes(shape_list)
{
    make2d(shapes);
    dim = shapes.size();
    strides = computeStrides(shapes);
}

//copy constructor
Tensor::Tensor(const Tensor& other)
    : valvec(other.valvec), shapes(other.shapes), strides(other.strides), dim(other.dim)
{
    if(!valvec.empty())
        basePtr = valvec.data();
    else
        basePtr = other.basePtr;
}

Tensor& Tensor::operator= (const Tensor& other){
    if(this != &other){
        valvec = other.valvec;
        shapes = other.shapes;
        strides = other.strides;
        dim = other.dim;

        if(!valvec.empty())
            basePtr = valvec.data();
        else
            basePtr = other.basePtr;
    }
    return *this;
}

std::tuple<std::vector<size_t>, std::vector<size_t>> Tensor::axis_reduction(const size_t axis){
    if(axis >= dim) {
            throw std::invalid_argument("Invalid axis value");
    }
    std::vector<size_t> reduced_dim;
    for(size_t i=0; i<dim; i++){
        if(i != axis){
            reduced_dim.push_back(i);
        }
    }

    std::vector<size_t> reduced_shape;
    for(size_t i=0; i<reduced_dim.size(); i++){
        reduced_shape.push_back(shapes[reduced_dim[i]]);
    }

    auto indices = NDRange(reduced_shape);
    
    std::vector<size_t> base_idx;
    for(const auto& idx: indices){
        size_t base = 0;
        for(size_t i=0;i<reduced_dim.size(); i++){
            base += idx[i]*strides[reduced_dim[i]];
        }
        base_idx.push_back(base);
    }
    make2d(reduced_shape, axis);
    return {base_idx, reduced_shape};
}

std::vector<size_t> Tensor::index(const size_t idx){
    std::vector<size_t> idx_vec(dim);
    size_t temp = idx;
    for(int d = (int)dim-1; d >= 0; d--){
        idx_vec[d] = temp% shapes[d];
        temp /= shapes[d];
    }
    return idx_vec;
}
 
size_t Tensor::jumpTo(std::vector<size_t> pos) const {
    if(pos.size() != dim) throw std::invalid_argument("Invalid size");
    size_t val_pos = 0;
    for(size_t i=0; i<dim; i++){
        val_pos += strides[i] * pos[i];
    }
    return val_pos;
}

const std::vector<size_t> Tensor::computeStrides(std::vector<size_t> shps) const{
    size_t p = 1;
    std::vector<size_t> std(dim);
    for(int i=dim-1; i>=0; i--){
        std[i] = p;
        p*=shps[i];
    }
    return std;
}

double* Tensor::data() { return basePtr; }
const double* Tensor::data() const { return basePtr; }

const std::vector<double> Tensor::as_vector_const(){ 
    if(!valvec.empty())
        return valvec; 

    size_t total_size = size();

    std::vector<double> res;
    auto indices = NDRange(shapes);
    for(const std::vector<size_t>& idx: indices){
        res.push_back(this->at(idx));
    }
    return res;
}

const bool Tensor::is_contiguous() const {
    std::vector<size_t> expected = computeStrides(shapes);
    return strides == expected;
}

std::vector<double>& Tensor::as_vector(){
    return this->valvec;
} 

size_t Tensor::ndim() const { return dim; }
std::vector<size_t> Tensor::get_strides() const { return strides; }    
const std::vector<size_t>& Tensor::shape() const { return shapes; }

size_t Tensor::size() const { return std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<size_t>()); }

bool Tensor::empty() const { return (dim == 0 ? true : false); }

Tensor Tensor::view(std::vector<size_t> new_shape){
    Tensor vw = *this;
    vw.basePtr = this->basePtr;
    vw.shapes = new_shape;
    vw.strides = computeStrides(new_shape);
    return vw;
}

//region broadcasting rules
bool Tensor::shape_check(std::vector<size_t> t_shp){
    size_t tdim = t_shp.size();
    size_t maxl = std::max(tdim, dim);

    std::vector<size_t> smv(maxl,1);
    const auto& src = tdim >= dim ? shapes : t_shp;
    const auto& trg = tdim >= dim ? t_shp : shapes;
    size_t nd = tdim >= dim ? dim : tdim;
    for(size_t i=0; i<nd; i++){
        smv[maxl-1-i] = src[nd-1-i];
    }

    for(size_t i=0; i<maxl; i++)
        if(!(smv[maxl-i-1] == trg[maxl-i-1] || smv[maxl-i-1] == 1 || trg[maxl-i-1] == 1)) throw std::runtime_error("Shapes mismatch"+vec_string(shapes)+" vs "+vec_string(t_shp));
    return true;
}

std::vector<size_t> Tensor::broadcast_shape(const Tensor& t){
    if(!shape_check(t.shape())) {}
    std::vector<size_t> fnl_shape;
    size_t maxl = std::max(t.ndim(), dim);

    std::vector<size_t>smv(maxl,1);
    const auto& src = t.ndim() >= dim ? shapes : t.shape();
    const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
    size_t nd = t.ndim() >= dim ? dim : t.ndim();
    for(size_t i=0; i<nd; i++){
        smv[maxl-1-i] = src[nd-1-i];
    }

    for(size_t i=0; i<maxl; i++)
        smv[maxl-i-1] = std::max(smv[maxl-i-1], trg[maxl-i-1]); 
    fnl_shape = smv;  
    return fnl_shape;
}

Tensor Tensor::singleton_rule(const Tensor& t){
    if(!shape_check(t.shape())) {}
    std::vector<size_t> fnl_shape;
    size_t maxl = std::max(t.ndim(), dim);

    std::vector<size_t>smv(maxl,1);
    const auto& src = t.ndim() >= dim ? shapes : t.shape();
    const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
    double* ptr = t.ndim() >= dim ? basePtr : const_cast<double*>(t.data());
    std::vector<size_t> newStrides = t.ndim() >= dim ? get_strides() : t.get_strides();
    size_t nd = t.ndim() >= dim ? dim : t.ndim();
    for(size_t i=0; i<nd; i++){
        smv[maxl-1-i] = src[nd-1-i];
    }
    newStrides.insert(newStrides.begin(), maxl-nd, 0);
    for(size_t i=0; i<maxl; i++){
        if(trg[maxl - i - 1]>smv[maxl - i - 1]){
            smv[maxl-i-1] = trg[maxl-i-1];
            newStrides[maxl-i-1] = 0;
        }
    }
    
    return Tensor(ptr, smv, newStrides);
}

Tensor Tensor::unsqueeze(size_t axis){
    Tensor view = *this;
    std::vector<size_t> new_shape = shapes;
    std::vector<size_t> new_strides = strides;
    new_shape.insert(new_shape.begin()+axis, 1);
    size_t new_stride = axis < strides.size() ? strides[axis] : 1;
    new_strides.insert(new_strides.begin()+axis, new_stride);
    
    return Tensor(basePtr, new_shape, new_strides);
}

Tensor Tensor::expand(std::vector<size_t> target){
    Tensor view = *this;
    std::vector<size_t> new_strides = strides;
    if(shape_check(target)){
        for(size_t i=0; i<target.size(); i++){
            if(shapes[i] > target[i]) throw std::runtime_error("Target shape should be greater");
            if(shapes[i] == 1 && target[i] == 1) continue;
            if(shapes[i] == 1) new_strides[i] = 0;
        }
    }
    // view.shapes = target;
    // view.strides = new_strides;
    // return view;
    return Tensor(basePtr, target, new_strides);
}


Tensor Tensor::mask_filled(const Tensor& mask, double replace){
    if(mask.size() != size()) throw std::invalid_argument("Mask/Matrix size mismatch");
    std::vector<double> n_vec(size());
   
    const double* m_ptr = mask.data();

    #pragma omp parallel for simd schedule(static) 
    for(int i=0; i<size(); i++){
        double val = mask.at(index(i));
        n_vec[i] = (val == 0.0 ? replace : val);
    }
    return Tensor(n_vec, shapes);
}

Tensor Tensor::replace_zero(const Tensor& other){
    if(shapes != other.shapes) throw std::invalid_argument("The 2 matrix shapes are incompatible for replacement" + vec_string(shapes) + " v " + vec_string(other.shapes));

    std::vector<double> n_vec(size());

    #pragma omp parallel for simd schedule(static)
    for(int i=0; i<size(); i++){
        std::vector<size_t> idx = index(i);
        n_vec[i] = this->at(idx) == 0.0 ? other.at(idx) : this->at(idx);
    }

    return Tensor(n_vec, shapes);
}

//region access and modification
double Tensor::at(const std::vector<size_t> pos) const {
    size_t val_pos = jumpTo(pos);
    return basePtr[val_pos];
}

void Tensor::put(std::vector<size_t> pos, double val) {
    size_t idx = jumpTo(pos);
    if(idx < size()) basePtr[idx] = val;
}
//endregion access and modification

//region data-viewing
// referenced slicing -> the slice is still pointing to the same location as the og tensor
Tensor Tensor::slice(std::vector<size_t> start, std::vector<size_t> shape, const std::optional<std::vector<size_t>>& _strides){
    std::vector<size_t> actualStrides = _strides.value_or(strides);
    Tensor slc = *this;
    slc.basePtr = this->basePtr + jumpTo(start);
    slc.shapes = shape;
    slc.strides = actualStrides;
    return slc;
}

Tensor Tensor::slice(size_t start, size_t end, std::vector<size_t> shape_list){
    std::vector<double> sliced_vec(valvec.begin()+start, valvec.begin()+end);
    return Tensor(sliced_vec, shape_list);
}

Tensor Tensor::reshape(std::vector<size_t> new_shape){
    size_t p = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if(size() != p) return Tensor();
    std::vector<size_t> new_strides = computeStrides(new_shape);
    return Tensor(basePtr, new_shape, new_strides);
}

Tensor Tensor::permute(const std::optional<std::vector<size_t>>& rotaxis) {
    std::vector<size_t> new_shape = shapes;    
    std::vector<size_t> new_strides = strides;
    for(size_t i=0; i<dim; i++){
        size_t axis = rotaxis.has_value() ? rotaxis->at(i) : dim - i - 1;
        new_shape[i] = shapes[axis];
        new_strides[i] = shapes[axis];
    }
    return Tensor(this->basePtr, new_shape, new_strides);
}

Tensor Tensor::transpose(){
    return permute();
}

std::vector<Tensor> Tensor::split_uneven(const std::vector<size_t>& split_len, const size_t axis){
    if(axis < 0 || axis >= shapes.size()) throw std::invalid_argument("Axis value is not a valid shape index - Shape of matrix: "+vec_string(shapes));
    
    std::vector<Tensor> splits;
    size_t offset = 0;

    for(size_t len: split_len){
        std::vector<size_t> op_coords(this->shapes.size(), 0);
        op_coords[axis] = offset;
        double* split_ptr = this->basePtr + this->jumpTo(op_coords);

        std::vector<size_t> split_shape = this->shapes;
        split_shape[axis] = len;

        splits.emplace_back(split_ptr, split_shape, this->strides);
        offset += len;
    }
    return splits;
}

std::vector<Tensor> Tensor::chunk(const size_t num_chunks, const size_t axis) {
    if(axis < 0 || axis >= shapes.size()) throw std::invalid_argument("Axis value is not a valid shape index - Shape of matrix: "+vec_string(shapes));
    
    size_t dim_size = this->shapes[axis];
    size_t chunk_size = (dim_size + num_chunks - 1) / num_chunks;

    std::vector<size_t> split_sizes;
    size_t left = dim_size;

    for(size_t i=0; i<num_chunks; i++){
        size_t current_size = std::min(chunk_size, left);
        if(current_size > 0){
            split_sizes.push_back(current_size);
            left -= current_size;
        }
    }

    return this->split_uneven(split_sizes, axis);
}
//endregion data-viewing

//region element-wise operations
Tensor Tensor::operator+ (const Tensor& t) {
    return tensorOp(t, [](double a, double b){ return a+b; });
}

Tensor Tensor::operator+ (double val){
    return scalarOp(val, [](double a, double b){ return a+b; }); 
}

Tensor Tensor::operator- (const Tensor& t){
    return tensorOp(t, [](double a, double b){ return a-b; });
}

Tensor Tensor::operator- (double val) {
    return scalarOp(val, [](double a, double b){ return a-b; });
}

Tensor Tensor::operator* (const Tensor& t) {
    return tensorOp(t, [](double a, double b){ return a*b; });
}

Tensor Tensor::operator* (double val) {
        return scalarOp(val, [](double a, double b){ return a*b; });
}

Tensor Tensor::operator/ (double val) {
    return scalarOp(val, [](double a, double b){ return a/b; });
}

Tensor Tensor::operator/ (const Tensor& t){
    return tensorOp(t, [](double a, double b){ return a/b; });
}

Tensor Tensor::operator+= (const Tensor& t){
    if(this->shape() != t.shape()) throw std::invalid_argument("Inplace addition matrix shapes incompatible");

    size_t total_size = this->size();

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        std::vector<size_t> idx = this->index(i);
        this->basePtr[this->jumpTo(idx)] += t.at(idx);
    }
    
    return *this;
}

//endregion element-wise operations

//region reductions
Tensor Tensor::sum(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<base_idx.size(); i++){
        double sum = 0;
        for(size_t j=0; j<shapes[axis]; j++){
            sum += basePtr[base_idx[i] + strides[axis]*j];
        }
        result[i] = sum;
    }

    return Tensor(result, reduced_shape);
}

Tensor Tensor::mean(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double sum = 0;
        for(size_t j=0; j<shapes[axis]; j++){
            sum += basePtr[base_idx[i] + strides[axis]*j];
        }
        result[i] = sum/shapes[axis];
    }
    return Tensor(result, reduced_shape);
}

Tensor Tensor::maximum(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double mx = -std::numeric_limits<double>::infinity();
        for(size_t j=0; j<shapes[axis]; j++){
            double val = basePtr[base_idx[i] + strides[axis]*j];
            mx = mx < val ? val : mx;
        }
        result[i] = mx;
    }
    return Tensor(result, reduced_shape);
}

Tensor Tensor::minimum(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double mn = std::numeric_limits<double>::infinity();
        for(size_t j=0; j<shapes[axis]; j++){
            double val = basePtr[base_idx[i] + strides[axis]*j];
            mn = mn > val ? val : mn;
        }
        result[i] = mn;
    }
    return Tensor(result, reduced_shape);
}
//endregion reductions

//element-wise functions
Tensor Tensor::sqrt() {
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::sqrt(this->at(index(i)));
    }

    return Tensor(res, shapes);
}

Tensor Tensor::log(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::log(this->at(index(i)));
    }

    return Tensor(res, shapes);
}

Tensor Tensor::exp(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::exp(this->at(index(i)));
    }

    return Tensor(res, shapes);
}

Tensor Tensor::pow(const double n){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::pow(this->at(index(i)), n);
    }

    return Tensor(res, shapes);

}
//functional operations

// Softmax function
Tensor Tensor::softmax(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(size());

    const size_t sax = shapes[axis];
    const size_t stride = strides[axis];

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double mx = -std::numeric_limits<double>::infinity();
        for(size_t j=0; j<sax; j++){
            double val = basePtr[base_idx[i] + stride*j];
            mx = mx < val ? val : mx;
        }
        double denom = 0;
        for(size_t j=0; j<sax; j++){
            denom += std::exp(basePtr[base_idx[i] + stride*j] - mx);
        }
        for(size_t j=0; j<sax; j++){
            size_t logi_idx = i*sax+j;
            size_t idx = base_idx[i] + stride*j;
            result[logi_idx] = std::exp(basePtr[idx] - mx) / denom;
        }
    }
    return Tensor(result, shapes);
}

Tensor Tensor::log_softmax(const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(size());

    const size_t sax = shapes[axis];
    const size_t stride = strides[axis];

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double mx = -std::numeric_limits<double>::infinity();
        for(size_t j=0; j<sax; j++){
            double val = basePtr[base_idx[i] + stride*j];
            mx = mx < val ? val : mx;
        }
        double denom = 0;
        for(size_t j=0; j<sax; j++){
            denom += std::exp(basePtr[base_idx[i] + stride*j] - mx);
        }

        for(size_t j=0; j<sax; j++){
            size_t logi_idx = i*sax + j;
            size_t idx = base_idx[i] + stride*j;
            result[logi_idx] = basePtr[idx] - mx - std::log(denom);
        }
    }
    return Tensor(result, shapes);
}

// Layer Normalization
Tensor Tensor::layer_norm(Tensor gamma, Tensor beta, const size_t axis){
    auto reductions = axis_reduction(axis);
    const std::vector<size_t>& base_idx = std::get<0>(reductions);
    const std::vector<size_t>& reduced_shape = std::get<1>(reductions);
    std::vector<double> result(size());

    const size_t sax = shapes[axis];
    const size_t stride = strides[axis];

    #pragma omp parallel for
    for(size_t i=0; i<base_idx.size(); i++){
        double mu = 0, var = 0;
        for(size_t j=0; j<sax; j++){
            mu += basePtr[base_idx[i] + stride*j];
        }
        mu /= sax;
        for(size_t j=0; j<sax; j++){
            var += std::pow((basePtr[base_idx[i] + stride*j]-mu),2);
        }
        var /= sax;
        double e = 1e-12;
        double inv_std = 1.0 / std::sqrt(var + e);
        for(size_t j=0; j<sax; j++){
            size_t logi_idx = i*sax + j;
            double val = basePtr[base_idx[i] + stride*j];
            double normalized = (val-mu) * inv_std;
            result[logi_idx] = normalized * gamma.data()[i] + beta.data()[j];
        }
    }

    return Tensor(result, shapes);
}

Tensor Tensor::relu(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i = 0; i<size(); i++){
        res[i] = std::max(this->at(index(i)), 0.0);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::gelu(){
    std::vector<double> res(size());

    const double constant = std::sqrt(2.0 / std::numbers::pi);

    #pragma omp parallel for simd schedule(static)
    for(size_t i = 0; i<size(); i++){
        double x = this->at(index(i));
        double cube = x*x*x;
        double inner = constant*(x+0.044715*cube);
        double val = 0.5 * x * (1.0 + std::tanh(inner));
        res[i] = val;
    }

    return Tensor(res, shapes);
}

Tensor Tensor::sigmoid(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i = 0; i<size(); i++){
        double val = this->at(index(i));
        res[i] = 1/(1+std::exp(-val));
    }
    
    return Tensor(res, shapes);
}

Tensor Tensor::tanh(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i = 0; i<size(); i++){
        std::vector<size_t> idx = index(i);
        res[i] = std::tanh(this->at(idx));
    }

    return Tensor(res, shapes);
}

Tensor Tensor::elu(const double alpha){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i = 0; i<size(); i++){
        std::vector<size_t> idx = index(i);
        double val_pos = this->at(idx);
        double val = val_pos > 0.0 ? val_pos : alpha * (std::exp(val_pos-1));
        res[i] = val;
    }

    return Tensor(res, shapes);
}

void Tensor::xavier_ud(const size_t fan_in, const size_t fan_out){
    double limit = std::sqrt(6.0 / (double)(fan_in + fan_out));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-limit, limit);

    for(size_t i=0; i<size(); i++){
        this->basePtr[i] = dist(gen);
    }
}

Tensor Tensor::dropout(const double p, const bool training, Tensor& mask){
    if(!training) return *this;

    std::vector<double> mask_vec(size());
    double scale = 1.0/(1.0 - p);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - p);

    for(size_t i=0; i<size(); i++){
        mask_vec[i] = dist(gen) ? scale : 0.0;
    }

    mask = Tensor(mask_vec, shape());
    return *this * mask;
}

//test operations
void Tensor::show(){
    for(int i=0; i<size(); i++)
        std::cout<<*(basePtr+i)<<" ";
}

void Tensor::prnt(std::vector<size_t> x){
    for(auto e: x){
        std::cout<<e<<" ";
    }
    std::cout<<std::endl;
}

void Tensor::prntd(std::vector<double> x){
    for(auto e: x){
        std::cout<<e<<" ";
    }
    std::cout<<std::endl;
}

void Tensor::make2d(std::vector<size_t>& shape_list, const size_t axis){
    if(shape_list.size() == 1 && axis == 1)
        shape_list.push_back(1);
    if(shape_list.size() == 1 && axis == 0)
        shape_list.insert(shape_list.begin(), 1);
}

Tensor concatenate(const std::vector<Tensor>& tensor_list, const size_t axis){
    std::vector<size_t> conc_shape = tensor_list[0].shape();
    
    for(size_t idx=0; idx<tensor_list.size(); idx++){
        if(conc_shape.size() != tensor_list[idx].ndim()) throw std::invalid_argument("Tensor dimension mismatch");
    }

    if(axis > conc_shape.size()) throw std::invalid_argument("Axis is not a valid dimension");

    for(size_t idx=1; idx<tensor_list.size(); idx++){
        for(size_t i=0; i<conc_shape.size(); i++){
            if(i != axis && conc_shape[i] != tensor_list[idx].shape()[i]) {
                throw std::runtime_error("Internal tensor shapes mismatch"+vec_string(conc_shape)+", "+vec_string(tensor_list[idx].shape()));
            }
        }
        conc_shape[axis] += tensor_list[idx].shape()[axis];
    }

    Tensor conc(conc_shape);
    
    double* conc_ptr = conc.data();

    size_t left_size = 1;
    size_t right_size = 1;

    for(size_t i=0; i<axis; i++){
        left_size *= conc_shape[i];
    }

    for(size_t i=axis+1; i<conc_shape.size(); i++){
        right_size *= conc_shape[i];
    }

    for(size_t left = 0; left < left_size; left++){
        for(const Tensor& t: tensor_list){
            size_t axis_val = t.shape()[axis];
            size_t chunk_size = axis_val * right_size;
            
            const double* t_ptr = t.data() + (left*chunk_size);

            std::memcpy(conc_ptr, t_ptr, chunk_size * sizeof(double));
            conc_ptr += chunk_size;
        }
    }
    
    return conc;
}

Tensor dot(Tensor x, Tensor y, const size_t axis){
    Tensor b = (x*y).sum(axis).unsqueeze(axis);
    return b;
}
