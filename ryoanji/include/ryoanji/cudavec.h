#pragma once

template<typename T>
class cudaVec {
private:
  bool PIN;
  int SIZE;
  T * HOST;
  T * DEVC;

  void dealloc() {
    if( SIZE != 0 ) {
      SIZE = 0;
      if (PIN) CUDA_SAFE_CALL(cudaFreeHost(HOST));
      CUDA_SAFE_CALL(cudaFree(DEVC));
    }
  }

public:
  cudaVec() : PIN(false), SIZE(0), HOST(NULL), DEVC(NULL) {}
  cudaVec(int size, bool pin=false) : PIN(pin), SIZE(size) {
    if (PIN) CUDA_SAFE_CALL(cudaMallocHost(&HOST, SIZE*sizeof(T), cudaHostAllocMapped || cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc(&DEVC, SIZE*sizeof(T)));
  }
  ~cudaVec() {
    dealloc();
  }

  void alloc(int size, bool pin=false) {
    dealloc();
    PIN = pin;
    SIZE = size;
    if (PIN) CUDA_SAFE_CALL(cudaMallocHost(&HOST, SIZE*sizeof(T), cudaHostAllocMapped || cudaHostAllocWriteCombined));
    CUDA_SAFE_CALL(cudaMalloc(&DEVC, SIZE*sizeof(T)));
  }

  void resize(int size) {
    SIZE = size;
  }

  void zeros() {
    CUDA_SAFE_CALL(cudaMemset(DEVC, 0, SIZE*sizeof(T)));
  }

  void ones() {
    CUDA_SAFE_CALL(cudaMemset(DEVC, 1, SIZE*sizeof(T)));
  }

  void d2h() {
    assert(PIN);
    CUDA_SAFE_CALL(cudaMemcpy(HOST, DEVC, SIZE*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void d2h(int size) {
    assert(PIN);
    CUDA_SAFE_CALL(cudaMemcpy(HOST, DEVC, size*sizeof(T), cudaMemcpyDeviceToHost));
  }

  void h2d() {
    assert(PIN);
    CUDA_SAFE_CALL(cudaMemcpy(DEVC, HOST, SIZE*sizeof(T), cudaMemcpyHostToDevice));
  }

  void h2d(int size) {
    assert(PIN);
    CUDA_SAFE_CALL(cudaMemcpy(DEVC, HOST, size*sizeof(T), cudaMemcpyHostToDevice));
  }

  template<typename S>
  void bind(texture<S,1,cudaReadModeElementType> &tex) {
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode     = cudaFilterModePoint;
    tex.normalized     = false;
    CUDA_SAFE_CALL(cudaBindTexture(0, tex, (S*)DEVC, SIZE*sizeof(T)));
  }

  template<typename S>
  void unbind(texture<S,1,cudaReadModeElementType> &tex) {
    CUDA_SAFE_CALL(cudaUnbindTexture(tex));
  }

  T& operator[] (int i) const { return HOST[i]; }
  T* h() const { return HOST; }
  T* d() const { return DEVC; }
  int size() const { return SIZE; }
};
