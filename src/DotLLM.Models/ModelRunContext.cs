using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models;

/// <summary>
/// Represents the context required to execute a model run. This includes the loaded model,
/// the associated GGUF file containing tensor data, and the configuration details of the model.
/// </summary>
public record ModelRunContext(IModel Model, GgufFile Gguf, ModelConfig Config);
