# 跨平台统一运行时 C 语言接口 InfiniRT & 微型 C++ 大模型推理引擎 InfiniInfer

## 使用方式

### 配置 XMake

- 配置 XMake（选择硬件平台）

  ```shell
  xmake f [--nv-gpu/--ascend-npu]=true -cv
  ```

- 只编译运行时库，不使用多卡通信以及模型推理引擎（默认为打开）

  ```shell
  xmake f --ccl=false --infer=false -cv
  ```

### 编译和部署

- 设置 `INFINI_ROOT` 环境变量（推荐，默认安装地址为 `$HOME/.infini`）

  ```shell
  export INFINI_ROOT=$HOME/.infini
  export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
  ```

- 编译和部署

  ```shell
  xmake && xmake install
  ```

### 测试推理引擎

- 需要先编译和部署运行时和推理引擎

  ```shell
  python test/model/test_llama.py --cuda path/to/model/dir/
  ```
