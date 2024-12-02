add_rules("mode.debug", "mode.release")

add_includedirs("include")

option("omp")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable OpenMP support")
option_end()

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Nvidia GPU functions")
    add_defines("ENABLE_NV_GPU")
option_end()


option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Cambricon MLU functions")
    add_defines("ENABLE_CAMBRICON_MLU")
option_end()

option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Ascend NPU functions")
    add_defines("ENABLE_ASCEND_NPU")
option_end()

option("ccl")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable multi-device communication support")
option_end()

option("infer")
    set_default(true)
    set_showmenu(true)
    set_description("Enable or disable end-to-end inference support")
    add_includedirs(os.getenv("INFINI_ROOT") .. "/lib/include")
option_end()

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end


if has_config("nv-gpu") then
    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
        on_install(function (target) end)
        set_policy("build.cuda.devlink", true)

        set_toolchains("cuda")
        add_links("cudart")

        if is_plat("windows") then
            add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        else
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
        end

        set_languages("cxx17")
        add_files("src/runtime/cuda/*.cc")
        if has_config("ccl") then
            -- Check if NCCL_ROOT is defined
            local nccl_root = os.getenv("NCCL_ROOT")
            if nccl_root then
                add_includedirs(nccl_root .. "/include")
                add_links(nccl_root .. "/lib/libnccl.so")
            else
                add_links("nccl") -- Fall back to default nccl linking
            end
            add_files("src/ccl/cuda/*.cc")
        end
    target_end()
end

if has_config("ascend-npu") then

    add_defines("ENABLE_ASCEND_NPU")
    local ASCEND_HOME = os.getenv("ASCEND_HOME")
    local SOC_VERSION = os.getenv("SOC_VERSION")

    -- Add include dirs
    add_includedirs(ASCEND_HOME .. "/include")
    add_includedirs(ASCEND_HOME .. "/include/aclnn")
    add_linkdirs(ASCEND_HOME .. "/lib64")
    add_links("libascendcl.so")
    add_links("libnnopbase.so")
    add_links("libopapi.so")
    add_links("libruntime.so")  
    add_linkdirs(ASCEND_HOME .. "/../../driver/lib64/driver")
    add_links("libascend_hal.so")
    

    target("ascend-npu")
        -- Other configs
        set_kind("static")
        set_languages("cxx17")
        on_install(function (target) end)
        -- Add files
        add_files("src/runtime/ascend/*.cc")
        if has_config("ccl") then
            add_includedirs(ASCEND_HOME .. "/include/hccl")
            add_links("libhccl.so")
            add_files("src/ccl/ascend/*cc")
        end
        add_cxflags("-lstdc++ -Wall -Werror -fPIC")

    target_end()
end

target("infinirt")
    set_kind("shared")

    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end

    set_languages("cxx17")
    add_files("src/runtime/runtime.cc")
    on_install(function (target) 
        os.cp(target:targetfile(), os.getenv("INFINI_ROOT") .. "/lib/libinfinirt.so")
    end)
target_end()

if has_config("ccl") then
target("infiniccl")
    set_kind("shared")

    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end
    set_languages("cxx17")
    add_files("src/ccl/infiniccl.cc")
    on_install(function (target) 
        os.cp(target:targetfile(), os.getenv("INFINI_ROOT") .. "/lib/libinfiniccl.so")
    end)
target_end()
end

if has_config("infer") then
target("infiniinfer")
    set_kind("shared")
    add_deps("infinirt")
    add_deps("infiniccl")
    add_links(os.getenv("INFINI_ROOT") .. "/lib/libinfiniop.so")
    set_languages("cxx17")
    add_files("src/models/*.cc")
    add_files("src/tensor/*.cc")
    add_includedirs("src")
    on_install(function (target) 
        os.cp(target:targetfile(), os.getenv("INFINI_ROOT") .. "/lib/libinfiniinfer.so")
    end)
target_end()
end

if has_config("infer") then
target("infini_infer_test")
    set_kind("binary")
    set_languages("cxx17")
    on_install(function (target) end)
    add_includedirs("src")
    if has_config("nv-gpu") then
        add_deps("nv-gpu")   
    end
    if has_config("ascend-npu") then
        add_deps("ascend-npu")
    end
    add_cxflags("-g", "-O0")
    add_ldflags("-g") 
    add_files("test/test.cc")
    add_files("test/tensor/*.cc")
    add_files("src/runtime/runtime.cc")
    if has_config("ccl") then
        add_files("src/ccl/infiniccl.cc")
        add_files("test/ccl/*.cc")
    end
    
    add_files("src/models/*.cc")
    add_links(os.getenv("INFINI_ROOT") .. "/lib/libinfiniop.so")
    add_files("src/tensor/*.cc")
    add_cxflags("-lstdc++ -Wall -fPIC")
    
    if has_config("omp") then
        add_cxflags("-fopenmp")
        add_ldflags("-fopenmp")
    end
    
target_end()
end
