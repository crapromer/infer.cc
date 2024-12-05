# 跨平台统一运行时C语言接口InfiniRT & 微型C++大模型推理引擎InfiniInfer

## 使用方式

### 配置XMake

* 配置XMake（选择硬件平台）

```
xmake f [--nv-gpu/--ascend-npu]=true -cv
```

* 只编译运行时库，不使用多卡通信以及模型推理引擎（默认为打开）

```
xmake f --ccl=false --infer=false -cv
```

### 编译和部署

* 设置INFINI_ROOT环境变量（推荐，默认安装地址为$HOME/.infini）

```
export INFINI_ROOT=$HOME/.infini
export LD_LIBRARY_PATH=$INFINI_ROOT/lib:$LD_LIBRARY_PATH
```

* 编译和部署
```
xmake && xmake install
```

### 测试推理引擎

* 需要先编译和部署运行时+推理引擎

```
python test/model/test_llama.py --cuda path/to/model/dir/
```
