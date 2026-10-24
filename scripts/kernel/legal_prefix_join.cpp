// Exact Jaccard >=4/5 prefix join. Every rank is an exact five-word tuple.
// O(prefix visits + candidate merge comparisons); no candidate or work cap.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

template<class T> struct Mapping {
    int fd; size_t bytes; const T* data;
    explicit Mapping(const std::string& path, size_t count) : bytes(count*sizeof(T)) {
        fd=open(path.c_str(),O_RDONLY);
        if(fd<0) throw std::runtime_error("JOIN_INPUT_OPEN");
        struct stat s{};
        if(fstat(fd,&s) || size_t(s.st_size)!=bytes) throw std::runtime_error("JOIN_INPUT_SIZE");
        data=bytes ? static_cast<const T*>(mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0)) : nullptr;
        if(bytes && data==MAP_FAILED) throw std::runtime_error("JOIN_INPUT_MAP");
    }
    ~Mapping(){if(bytes) munmap(const_cast<T*>(data),bytes);close(fd);}
    const T& operator[](size_t i) const{return data[i];}
};

int main(int argc,char** argv){
  try {
    if(argc!=7) throw std::runtime_error("JOIN_ARGS: ranks offsets eligible documents rank_count output");
    auto start=std::chrono::steady_clock::now();
    const uint64_t n=std::stoull(argv[4]), rank_count=std::stoull(argv[5]);
    if(n>=UINT32_MAX || rank_count>=UINT32_MAX) throw std::runtime_error("JOIN_ID_WIDTH");
    Mapping<int64_t> offsets(argv[2],n+1);
    if(offsets[0]!=0 || offsets[n]<0) throw std::runtime_error("JOIN_OFFSET_RANGE");
    Mapping<uint32_t> ranks(argv[1],offsets[n]); Mapping<uint8_t> eligible(argv[3],n);
    std::vector<uint32_t> order;
    std::vector<uint64_t> lengths(n), prefix(n);
    for(uint32_t i=0;i<n;++i){
        if(offsets[i]<0 || offsets[i+1]<offsets[i] || eligible[i]>1) throw std::runtime_error("JOIN_OFFSETS");
        uint64_t m=lengths[i]=offsets[i+1]-offsets[i];
        for(int64_t j=offsets[i];j<offsets[i+1];++j)
            if(ranks[j]>=rank_count || (j>offsets[i] && ranks[j]<=ranks[j-1]))
                throw std::runtime_error("JOIN_RANK_ORDER");
        if(eligible[i]){
            if(!m) throw std::runtime_error("JOIN_EMPTY_ELIGIBLE");
            order.push_back(i); prefix[i]=m-(4*m+4)/5+1;
        }
    }
    std::sort(order.begin(),order.end(),[&](uint32_t a,uint32_t b){
        return lengths[a]!=lengths[b] ? lengths[a]<lengths[b] : a<b;
    });
    std::vector<uint32_t> frequency(rank_count,0);
    for(auto i:order) for(uint64_t j=0;j<prefix[i];++j) ++frequency[ranks[offsets[i]+j]];
    std::vector<uint64_t> starts(rank_count+1,0);
    for(uint64_t r=0;r<rank_count;++r) starts[r+1]=starts[r]+frequency[r];
    std::vector<uint32_t> postings(starts.back());
    std::fill(frequency.begin(),frequency.end(),0);
    for(auto i:order) for(uint64_t j=0;j<prefix[i];++j){
        auto r=ranks[offsets[i]+j]; postings[starts[r]+frequency[r]++]=i;
    }
    const auto indexed=std::chrono::steady_clock::now();
    std::vector<uint32_t> marked(n,UINT32_MAX);
    uint64_t visits=0,candidates=0,comparisons=0,edges=0;
    std::ofstream out(argv[6],std::ios::binary|std::ios::out|std::ios::trunc);
    if(!out) throw std::runtime_error("JOIN_OUTPUT_OPEN");
    for(auto i:order){
        const uint64_t m=lengths[i], min_size=(4*m+4)/5;
        for(uint64_t p=0;p<prefix[i];++p){
            const auto rank=ranks[offsets[i]+p];
            if(frequency[rank]<2) continue;
            auto begin=postings.begin()+starts[rank],end=postings.begin()+starts[rank+1];
            auto found=std::lower_bound(begin,end,min_size,[&](uint32_t d,uint64_t v){return lengths[d]<v;});
            for(;found!=end;++found){
                const auto j=*found;
                if(lengths[j]>m || (lengths[j]==m && j>=i)) break;
                ++visits;
                if(marked[j]==i) continue;
                marked[j]=i; ++candidates;
                uint64_t a=offsets[i],b=offsets[j],common=0;
                const uint64_t ae=offsets[i+1],be=offsets[j+1],needed=(4*(m+lengths[j])+8)/9;
                while(a<ae && b<be){
                    ++comparisons;
                    if(ranks[a]==ranks[b]){++common;++a;++b;}
                    else if(ranks[a]<ranks[b]) ++a;
                    else ++b;
                    if(common+std::min(ae-a,be-b)<needed) break;
                }
                if(9*common>=4*(m+lengths[j])){
                    if(common>UINT32_MAX) throw std::runtime_error("JOIN_INTERSECTION_WIDTH");
                    const std::array<uint32_t,3> edge{std::min(i,j),std::max(i,j),uint32_t(common)};
                    out.write(reinterpret_cast<const char*>(edge.data()),sizeof(edge));++edges;
                    if(!out) throw std::runtime_error("JOIN_OUTPUT_WRITE");
                }
            }
        }
    }
    out.flush(); if(!out) throw std::runtime_error("JOIN_OUTPUT_FLUSH"); out.close();
    auto finish=std::chrono::steady_clock::now();
    std::cout<<"{\"documents\":"<<n<<",\"eligible\":"<<order.size()
             <<",\"prefix_memberships\":"<<starts.back()<<",\"posting_visits\":"<<visits
             <<",\"candidates\":"<<candidates<<",\"merge_comparisons\":"<<comparisons
             <<",\"edges\":"<<edges<<",\"index_seconds\":"<<std::chrono::duration<double>(indexed-start).count()
             <<",\"join_seconds\":"<<std::chrono::duration<double>(finish-indexed).count()
             <<",\"seconds\":"<<std::chrono::duration<double>(finish-start).count()<<"}\n";
    return 0;
  } catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}
}
