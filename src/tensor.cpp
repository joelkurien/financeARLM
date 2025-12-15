#include "tensor.h"

Tensor::Tensor(std::vector<size_t> shape_list)
{
    if(shape_list.size() < 2) throw std::invalid_argument("Vector shape should have at least 2 dimensions");
    shapes = shape_list;
    dim = shapes.size();
    strides = computeStrides(shapes);
    valvec.resize(size(),0);
    basePtr = valvec.data();
}

Tensor::Tensor(std::vector<double> vec, std::vector<size_t> shape_list)
    : shapes(shape_list), dim(shapes.size()), valvec(vec)
{
    basePtr = valvec.data();
    strides = computeStrides(shapes);
}

Tensor::Tensor(double* ptr, std::vector<size_t> shape_list, std::vector<size_t> strides)
    : basePtr(ptr), strides(strides), shapes(shape_list), dim(shapes.size()) {}

//copy constructor
Tensor::Tensor(const Tensor& other)
    : valvec(other.valvec), shapes(other.shapes), strides(other.strides), dim(other.dim)
{
    basePtr = valvec.data();
}

Tensor& Tensor::operator= (const Tensor& other){
    if(this != &other){
        valvec = other.valvec;
        shapes = other.shapes;
        strides = other.strides;
        dim = other.dim;
        basePtr = valvec.data();
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

std::vector<double> Tensor::as_vector() { return valvec; }
size_t Tensor::ndim() const { return dim; }
std::vector<size_t> Tensor::get_strides() const { return strides; }    
std::vector<size_t> Tensor::shape() const { return shapes; }

size_t Tensor::size() const { return std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<size_t>()); }

bool Tensor::empty() const { return (dim == 0 ? true : false); }

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
        if(!(smv[maxl-i-1] == trg[maxl-i-1] || smv[maxl-i-1] == 1 || trg[maxl-i-1] == 1)) throw std::runtime_error("Shapes incompatible");
    return true;
}

std::vector<size_t> Tensor::broadcast_shape(Tensor& t){
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

Tensor Tensor::singleton_rule(Tensor& t){
    if(!shape_check(t.shape())) {}
    std::vector<size_t> fnl_shape;
    size_t maxl = std::max(t.ndim(), dim);

    std::vector<size_t>smv(maxl,1);
    const auto& src = t.ndim() >= dim ? shapes : t.shape();
    const auto& trg = t.ndim() >= dim ? t.shape() : shapes;
    double* ptr = t.ndim() >= dim ? basePtr : t.data();
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
    std::vector<size_t> new_shape = shapes;
    std::vector<size_t> new_strides = strides;
    new_shape.insert(new_shape.begin()+axis, 1);
    new_strides.insert(new_strides.begin()+axis, 0);
    return Tensor(basePtr, new_shape, new_strides);
}

Tensor Tensor::expand(std::vector<size_t> target){
    std::vector<size_t> new_strides = strides;
    if(shape_check(target)){
        for(size_t i=0; i<target.size(); i++){
            if(shapes[i] > target[i]) throw std::runtime_error("Target shape should be greater");
            if(shapes[i] == 1 && target[i] == 1) continue;
            if(shapes[i] == 1) new_strides[i] = 0;
        }
    }
    return Tensor(basePtr, target, new_strides);
}

Tensor Tensor::concatenate(Tensor& b, const size_t axis){
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

        std::copy(valvec.begin()+b_lft, valvec.begin()+b_lft+b_size, conc.valvec.begin()+conc_idx);
        conc_idx += b_size;
    }
    // vector<size_t> left (shapes.begin(), shapes.begin()+axis);

    // vector<size_t> r1 (shapes.begin()+axis, shapes.end());
    // vector<size_t> r2 (b.shapes.begin()+axis, b.shapes.end());
    // auto l_indices = NDRange(left);
    // auto a_indices = NDRange(r1);
    // auto b_indices = NDRange(r2);

    // Tensor conc (new_shape);
    // size_t index = 0;
    // for(const vector<size_t>& l_idx: l_indices){
    //     vector<size_t> pos (dim, 0);

    //     for(size_t i=0; i<l_idx.size(); i++){
    //         pos[i] = l_idx[i];
    //     }
    //     for(const vector<size_t>& idx: a_indices){
    //         for(size_t x=0; x<idx.size(); x++){
    //             pos[axis+x] = idx[x];
    //         }
    //         conc.valvec[index++] = at(pos);
    //     }
    //     for(const vector<size_t>& idx: b_indices){
    //         for(size_t x=0; x<idx.size(); x++){
    //             pos[axis+x] = idx[x];
    //         }
    //         conc.valvec[index++] = b.at(pos);
    //     }
    // }
    return conc;
}

Tensor Tensor::mask_filled(std::vector<bool> mask, double replace){
    if(mask.size() != size()) throw std::invalid_argument("Mask/Matrix size mismatch");
    std::vector<double> n_vec;
    n_vec.reserve(size());
    for(int i=0; i<size(); i++){
        n_vec.push_back(mask[i] ? replace : valvec[i]);
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
    double* subBasePtr = basePtr + jumpTo(start);
    return Tensor(subBasePtr, shape, actualStrides);
}

Tensor Tensor::reshape(std::vector<size_t> new_shape){
    size_t p = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<size_t>());
    if(size() != p) return Tensor();
    std::vector<size_t> new_strides = computeStrides(new_shape);
    return Tensor(basePtr, new_shape, new_strides);
}

Tensor Tensor::permute(const std::optional<std::vector<size_t>>& rotaxis) {
    std::vector<size_t> new_stride = strides;
    std::vector<size_t> new_shape = shapes;
    if(rotaxis != std::nullopt && rotaxis->size() == dim){
        for(size_t i=0; i<dim; i++){
            new_stride[i] = strides[rotaxis->at(i)];
            new_shape[i] = shapes[rotaxis->at(i)];
        }
    } else {
        std::reverse(new_stride.begin(), new_stride.end());
        std::reverse(new_shape.begin(), new_shape.end());
    }

    return Tensor(basePtr, new_shape, new_stride);  
}
//endregion data-viewing

//region element-wise operations
Tensor Tensor::operator+ (Tensor& t) {
    return tensorOp(t, [](double a, double b){ return a+b; });
}

Tensor Tensor::operator+ (double val){
    return scalarOp(val, [](double a, double b){ return a+b; }); 
}

Tensor Tensor::operator- (Tensor& t){
    return tensorOp(t, [](double a, double b){ return a-b; });
}

Tensor Tensor::operator- (double val) {
    return scalarOp(val, [](double a, double b){ return a-b; });
}

Tensor Tensor::operator* (Tensor& t) {
    return tensorOp(t, [](double a, double b){ return a*b; });
}

Tensor Tensor::operator* (double val) {
        return scalarOp(val, [](double a, double b){ return a*b; });
}

Tensor Tensor::operator/ (double val) {
    return scalarOp(val, [](double a, double b){ return a/b; });
}

Tensor Tensor::operator/ (Tensor& t){
    return tensorOp(t, [](double a, double b){ return a/b; });
}
//endregion element-wise operations

//region reductions
Tensor Tensor::sum(const size_t axis){
    auto [base_idx, reduced_shape] = axis_reduction(axis);
    std::vector<double> result(base_idx.size());

    #pragma omp parallel for
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
            denom += exp(basePtr[base_idx[i] + stride*j] - mx);
        }
        for(size_t j=0; j<sax; j++){
            size_t idx = base_idx[i] + stride*j;
            result[idx] = exp(basePtr[idx] - mx) / denom;
        }
    }
    return Tensor(result, shapes);
}

// Layer Normalization
Tensor Tensor::layer_norm(const size_t gamma, const size_t beta, const size_t axis){
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
            var += pow((basePtr[base_idx[i] + stride*j]-mu),2);
        }
        var /= sax;
        double e = 1e-12;
        double inv_std = 1.0 / sqrt(var + e);
        for(size_t j=0; j<sax; j++){
            size_t idx = base_idx[i] + stride*j;
            result[idx] = gamma * ((basePtr[idx] - mu) * inv_std ) + beta;
        }
    }
    return Tensor(result, shapes);
}

Tensor Tensor::relu(){
    std::vector<double> res;
    res.reserve(size());

    for(int i=0; i<size(); i++){
        res.push_back(std::max(valvec[i], 0.0));
    }

    return Tensor(res, shapes);
}

Tensor Tensor::gelu(){
    std::vector<double> res;
    res.reserve(size());

    const double constant = sqrt(2.0 / std::numbers::pi);
    for(int i=0; i<size(); i++){
        double x = valvec[i];
        double cube = x*x*x;
        double inner = constant*(x+0.044715*cube);
        double val = 0.5 * x * (1.0 + tanh(inner));
        res.push_back(val);
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