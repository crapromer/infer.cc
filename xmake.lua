add_rules("mode.debug", "mode.release")

add_includedirs(os.getenv("INFINI_ROOT") .. "/include")

add_includedirs("include")


option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Nvidia GPU kernel")
    add_defines("ENABLE_NV_GPU")
option_end()


option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable Cambricon MLU kernel")
    add_defines("ENABLE_CAMBRICON_MLU")
option_end()


if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end


if has_config("nv-gpu") then
    add_defines("ENABLE_NV_GPU")
    target("nv-gpu")
        set_kind("static")
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
    target_end()
end


target("infinirt")
    set_kind("shared")

    if has_config("nv-gpu") then
        add_deps("nv-gpu")
    end

    set_languages("cxx17")
    add_files("src/runtime/runtime.cc")
target_end()


target("infini_infer")
    set_kind("shared")
    add_deps("infinirt")
    add_links(os.getenv("INFINI_ROOT") .. "/liboperators.so")
    set_languages("cxx17")
    add_files("src/models/*.cc")
    add_files("src/tensor/*.cc")
    add_includedirs("src")
target_end()

target("infini_infer_test")
    set_kind("binary")
    set_languages("cxx17")
    add_includedirs("src")
    if has_config("nv-gpu") then
        add_deps("nv-gpu")   
    end
    add_files("test/test.cc")
    add_files("test/tensor/*.cc")
    add_files("src/runtime/runtime.cc")
    add_files("src/models/*.cc")
    add_files("src/tensor/*.cc")
   
    add_links(os.getenv("INFINI_ROOT") .. "/liboperators.so")
    add_cxflags("-fopenmp")
    add_ldflags("-fopenmp")
    
target_end()
