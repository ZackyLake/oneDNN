/*******************************************************************************
* Copyright 2024-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/jit/generator.hpp"

#include "gpu/intel/jit/utils/utils.hpp"
#include "ngen_register_decl.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

    
struct Record
{
    std::thread::id Tid;
    std::string KernelName;
    std::vector<uint8_t> KernelBin;
    size_t BinHash = 0;
    float TimeMs = 0;
    primitive_kind_t Kind;
    Record(primitive_kind_t kind) noexcept : Tid(std::this_thread::get_id()), Kind(kind)  {}
    bool operator==(const Record & other) const noexcept { return KernelBin == other.KernelBin; }
};
RecordWrap::~RecordWrap() 
{
    const auto tend = std::chrono::high_resolution_clock::now();
    Rec.TimeMs = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(tend - TimeBegin).count();
    GetCurRecBin() = nullptr;
    if (!Rec.KernelBin.empty())
        Rec.BinHash = std::hash<std::string_view>{}(std::string_view { 
            reinterpret_cast<const char*>(Rec.KernelBin.data()), Rec.KernelBin.size() });
}
void RecordWrap::AssignName(std::string name) noexcept 
{
    Rec.KernelName = std::move(name);
}

struct NGenRecord 
{
    std::deque<Record> Records;
    std::atomic_flag Lock = ATOMIC_FLAG_INIT;
    const bool Dump = CheckEnv("dumpngenbin", "true");
    ~NGenRecord()
    {
        while (Lock.test_and_set());
        std::map<std::thread::id, TimeCounter> TidMap;
        struct RecordHasher
        {
            constexpr size_t operator()(const Record* rec) const noexcept
            {
                return rec->BinHash;
            }
        };
        using BinMap = std::unordered_map<const Record*, uint32_t, RecordHasher>;
        std::map<std::string_view, std::pair<BinMap, TimeCounter>> KerMap;
        TimeCounter total;
        for (const auto& rec : Records)
        {
            TidMap[rec.Tid].Add(rec.TimeMs);
            std::string_view kName = rec.KernelName;
            if (kName.empty()) kName = dnnl_prim_kind2str(rec.Kind);
            auto& [binmap, cnt] = KerMap[kName];
            if (auto [it, inserted] = binmap.try_emplace(&rec, 1); !inserted)
                it->second++;
            else if (!rec.KernelBin.empty() && Dump) 
            {
                std::string fname(".dump/");
                fname.append(kName).append("_").append(std::to_string(rec.BinHash)).append(".bin");
                std::ofstream stream(fname, std::ios::binary);
                stream.write(reinterpret_cast<const char*>(rec.KernelBin.data()), rec.KernelBin.size());
            }
            cnt.Add(rec.TimeMs);
            total.Add(rec.TimeMs);
        }
        const auto& maxTCnt = std::max_element(TidMap.begin(), TidMap.end(), [](const auto& lhs, const auto& rhs) { return lhs.second.Time > rhs.second.Time; })->second;
        printf("@@##NGen Kernels : [%u] in [%8.2f]ms(max [%7.2f]ms), [%zu] threads, max[%8.2f]ms\n", 
            total.Count, total.Time, total.MaxTime, TidMap.size(), maxTCnt.Time);
        for (const auto& [name, p] : KerMap)
        {
            const auto& [binmap, cnt] = p;
            const auto maxCnt = std::max_element(binmap.begin(), binmap.end(), [](const auto &lhs, const auto &rhs) { return lhs.second > rhs.second; })->second;
            printf("--[%-30s] : [%8.2f]ms(max [%7.2f]ms) @[%u] ([%zu] unique, max dup[%u])\n", name.data(), cnt.Time, cnt.MaxTime, cnt.Count, binmap.size(), maxCnt);
        }
    }
};
RecordWrap PutNGenRecord(primitive_kind_t kind) noexcept 
{
    static NGenRecord Records;
    while (Records.Lock.test_and_set());
    auto& rec = Records.Records.emplace_back(kind);
    Records.Lock.clear();
    GetCurRecBin() = &rec.KernelBin;
    return rec;
}


void check_kernel_size(const std::string &kernel_name, size_t kernel_size,
        size_t icache_size) {
    if (kernel_size > icache_size) {
        gpu_warning() << kernel_name
                      << " larger than icache, kernel: " << kernel_size
                      << " bytes, icache: " << icache_size << " bytes";
    }
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
