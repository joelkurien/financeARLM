#include "tensor.h"

#define THROW_ERR(msg) \
    throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
    " in function " + __func__ + "() -> " + msg)

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

 
size_t Tensor::jumpTo(std::vector<size_t> pos){
    if(pos.size() != dim) throw std::invalid_argument("Invalid size");
    size_t val_pos = 0;
    for(size_t i=0; i<dim; i++){
        val_pos += strides[i] * pos[i];
    }
    return val_pos;
}

std::vector<size_t> Tensor::computeStrides(std::vector<size_t> shps){
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

std::vector<double> Tensor::as_vector() { 
    if(!valvec.empty())
        return valvec; 

    size_t total_size = size();

    std::vector<double> res;
    res.reserve(total_size);

    std::vector<size_t> idxes(dim, 0);
    for(size_t i=0; i<total_size; i++){
        size_t offset = 0;
        for(size_t d=0; d<dim; d++){
            offset += idxes[d]*strides[d];
        }
        res.push_back(basePtr[offset]);

        for (int d = dim - 1; d >= 0; --d) {
            idxes[d]++;
            if (idxes[d] < shapes[d]) break;
            idxes[d] = 0;
        }
    }
    return res;
}
size_t Tensor::ndim() const { return dim; }
std::vector<size_t> Tensor::get_strides() const { return strides; }    
std::vector<size_t> Tensor::shape() const { return shapes; }

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
        if(!(smv[maxl-i-1] == trg[maxl-i-1] || smv[maxl-i-1] == 1 || trg[maxl-i-1] == 1)) THROW_ERR("Shapes mismatch"+vec_string(shapes)+" vs "+vec_string(t_shp));
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

    new_strides = computeStrides(new_shape);
    
    return Tensor(basePtr, new_shape);
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

Tensor Tensor::concatenate(const Tensor& b, const size_t axis){
    if(dim != b.ndim()) throw std::invalid_argument("Tensor dimension mismatch");
    if(axis >= dim) throw std::invalid_argument("Axis is invalid");
    std::vector<size_t> new_shape(dim,0);
    for(int i=0; i<dim; i++){
        if(i != axis && shapes[i] != b.shapes[i]){
            throw std::invalid_argument("Tensor non-target axis value mismatch");
        } 
        new_shape[i] = (i == axis) ? shapes[axis]+b.shapes[axis] : shapes[i];
    }

    Tensor conc(new_shape);

    size_t left_size = 1;
    for(size_t i=0; i<axis; i++)
        left_size *= shapes[i];
    
    size_t a_size = 1;
    for(size_t i = axis; i<dim; i++){
        a_size *= shapes[i];
    }

    size_t b_size = 1;
    for(size_t i = axis; i<dim; i++){
        b_size *= b.shapes[i];
    }

    size_t conc_idx = 0;
    for(size_t c_idx = 0; c_idx < left_size; c_idx++){
        size_t a_lft = c_idx*a_size;
        size_t b_lft = c_idx*b_size;

        std::copy(valvec.begin()+a_lft, valvec.begin() + a_lft + a_size, conc.valvec.begin()+conc_idx);
        conc_idx += a_size;

        std::copy(b.valvec.begin()+b_lft, b.valvec.begin()+b_lft+b_size, conc.valvec.begin()+conc_idx);
        conc_idx += b_size;
    }
    return conc;
}

Tensor Tensor::mask_filled(const Tensor& mask, double replace){
    if(mask.size() != size()) throw std::invalid_argument("Mask/Matrix size mismatch");
    std::vector<double> n_vec(size());
   
    const double* m_ptr = mask.data();

    #pragma omp parallel for simd schedule(static) 
    for(int i=0; i<size(); i++){
        n_vec[i] = (m_ptr[i] == 0.0 ? replace : basePtr[i]);
    }
    return Tensor(n_vec, shapes);
}

//region access and modification
double Tensor::at(std::vector<size_t> pos){
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
    for(size_t i=0; i<dim; i++){
        if(rotaxis.has_value() && rotaxis->size() == dim){
            new_shape[i] = this->shapes[rotaxis->at(i)];
        } else {
            new_shape[i] = this->shapes[dim-i-1];
        }
    }
    std::vector<size_t> new_strides = computeStrides(new_shape);
    return Tensor(this->basePtr, new_shape, new_strides);
}

Tensor Tensor::transpose(){
    return permute();
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
//endregion element-wise operations

//region reductions
Tensor Tensor::sum(const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<base_idx.size(); i++){
        double sum = 0;
        for(size_t j=0; j<shapes[axis]; j++){
            sum += basePtr[base_idx[i] + strides[axis]*j];
        }
        result[i] = sum;
    }

    if(reduced_shape.empty()) reduced_shape.push_back(1);
    return Tensor(result, reduced_shape);
}

Tensor Tensor::mean(const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
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
    auto [base_idx, reduced_shape] = axis_reduction(axis);
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
//endregion reductions

//element-wise functions
Tensor Tensor::sqrt() {
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::sqrt(valvec[i]);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::log(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::log(valvec[i]);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::exp(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::exp(valvec[i]);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::pow(const double n){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::pow(valvec[i], n);
    }

    return Tensor(res, shapes);

}
//functional operations

// Softmax function
Tensor Tensor::softmax(const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
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
            size_t idx = base_idx[i] + stride*j;
            result[idx] = std::exp(basePtr[idx] - mx) / denom;
        }
    }
    return Tensor(result, shapes);
}

Tensor Tensor::log_softmax(const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
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
            size_t idx = base_idx[i] + stride*j;
            result[idx] = basePtr[idx] - mx - std::log(denom);
        }
    }
    return Tensor(result, shapes);
}

// Layer Normalization
Tensor Tensor::layer_norm(Tensor gamma, Tensor beta, const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
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
            size_t idx = base_idx[i] + stride*j;
            result[idx] = (basePtr[idx] - mu) * inv_std;
        }
    }

    Tensor res = gamma * Tensor(result, shapes) + beta;
    return res;
}

Tensor Tensor::relu(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::max(valvec[i], 0.0);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::gelu(){
    std::vector<double> res(size());

    const double constant = std::sqrt(2.0 / std::numbers::pi);

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        double x = valvec[i];
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
    for(size_t i=0; i<size(); i++){
        res[i] = 1/(1+std::exp(-basePtr[i]));
    }
    
    return Tensor(res, shapes);
}

Tensor Tensor::tanh(){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = std::tanh(basePtr[i]);
    }

    return Tensor(res, shapes);
}

Tensor Tensor::elu(const double alpha){
    std::vector<double> res(size());

    #pragma omp parallel for simd schedule(static)
    for(size_t i=0; i<size(); i++){
        res[i] = basePtr[i] > 0.0 ? basePtr[i] : alpha * (std::exp(basePtr[i]-1));
    }

    return Tensor(res, shapes);
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


Tensor dot(Tensor x, Tensor y, const size_t axis){
    Tensor b = (x*y).sum(axis).unsqueeze(axis);
    return b;
}
