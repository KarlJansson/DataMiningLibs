#pragma once
#include <tbb/tbb.h>
#include <condition_variable>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T>
using sp = std::shared_ptr<T>;
template <typename T>
using wp = std::weak_ptr<T>;
template <typename T>
using up = std::unique_ptr<T>;
template <typename T, size_t Size>
using fixed_array = std::array<T, Size>;
template <typename T>
using col_array = std::vector<T, tbb::tbb_allocator<T>>;
template <typename T>
using col_list = std::list<T, tbb::tbb_allocator<T>>;
template <typename T, class comp = std::less<T>>
using col_set = std::set<T, comp, tbb::tbb_allocator<T>>;
template <typename K, typename V, class comp = std::less<K>>
using col_map = std::map<K, V, comp, tbb::tbb_allocator<std::pair<K, V>>>;
template <typename K, typename V>
using col_umap = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                                    tbb::tbb_allocator<std::pair<K, V>>>;
using string = std::string;
using mutex = std::mutex;
using mutex_lock = std::unique_lock<mutex>;
using condition_var = std::condition_variable;
