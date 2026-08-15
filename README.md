# Torch Model Architectures

Readable C++ implementations of common model architectures, built as native Torch module's.
Supports loading models from HuggingFace model directory and safetensors.

A diggestible set of files letting you easily match and find suitable architectures, small enough to read in one sitting and change when a model doesn't fit.

Currently implemented: 
- **Gemma 3**

## Use

```cpp
#include <modules/tokenizer.hpp>
#include <modules/model.hpp>

torch::Device device(torch::kCPU);
const char *path = "models/embeddinggemma-300m/"; // trailing slash

torch::nn::Sequential model = load_modules(path, device);
Tokenizer tokenize = load_tokenizer(path, device);
if (!model || !tokenize) {
  return 1; // both test like pointers, empty means loading failed
}

torch::Tensor embedding = model->forward(tokenize("hello world"));
```

### Dependencies

- [Libtorch](https://pytorch.org/cppdocs/installing.html)
- [nlohmann's json](https://github.com/nlohmann/json)

## License

MIT — see [LICENSE](LICENSE).

This project is not affiliated with, endorsed by, or sponsored by any model
provider. Architecture names are used only to identify which models these
implementations are compatible with. Model weights are not distributed here and
remain subject to their respective providers' licenses and terms of use.
