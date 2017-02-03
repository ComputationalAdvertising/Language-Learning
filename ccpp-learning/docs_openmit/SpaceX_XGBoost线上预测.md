## SpaceX: XGBoost线上预测

+ author: zhouyong

issue:

1. 如果利用xgb接口，需要线上环境支持c++11，好处是代码量少；否则，只能重写着部分代码。

## XGB数据加载

+ 数据处理逻辑

```c++
// 1. 初始化一个parser. 仍然是调用dmlc-core基础组件
std::unique_ptr<dmlc::Parser<uint32_t> > parser(
    dmlc::Parser<uint32_t>::Create(fname.c_str(), partid, npart, file_format.c_str()));
// 2. 利用parser创建一个DMatrix.
DMatrix* DMatrix::Create(dmlc::Parser<uint32_t>* parser, const std::string& cache_prefix) {};
// 2.1. 当cache_prefix.length == 0时：
std::unique_ptr<data::SimpleCSRSource> source(new data::SimpleCSRSource());
source->CopyFrom(parser);
// 2.1.1 是source->CopyFrom(parser)的具体实现
while (parser->Next()) {    const dmlc::RowBlock<uint32_t>& batch = parser->Value();
    ......
}
// 2.2. 返回结果
DMatrix::Create(std::move(source), cache_prefix);
// 2.2.1 是2.2的实现
DMatrix* DMatrix::Create(std::unique_ptr<DataSource>&& source, const std::string& cache_prefix) {};
// 2.2.2 用DataSource来初始化DMatrix 
new data::SimpleDMatrix(std::move(source));
```


## XGB模型结构

+ 模型相关数据结构整体如下：`~/src/gbm/gbtree.cc`

```c++
  // --- data structure ---  // base margin  float base_margin_;  // training parameter  GBTreeTrainParam tparam;  // model parameter  GBTreeModelParam mparam;  /*! \brief vector of trees stored in the model */  std::vector<std::unique_ptr<RegTree> > trees;  /*! \brief some information indicator of the tree, reserved */  std::vector<int> tree_info;  // ----training fields----  std::unordered_map<DMatrix*, CacheEntry> cache_;  // configurations for tree  std::vector<std::pair<std::string, std::string> > cfg;  // temporal storage for per thread  std::vector<RegTree::FVec> thread_temp;  // the updaters that can be applied to each of tree  std::vector<std::unique_ptr<TreeUpdater> > updaters;
```

+ 单颗树的结构：RegTree

```c++

```

## 预测打分

### 模型初始化

代码位置：`~/src/cli_main.cc`

```c++
std::unique_ptr<Learner> learner(Learner::Create({}));std::unique_ptr<dmlc::Stream> fi(dmlc::Stream::Create(param.model_in.c_str(), "r"));learner->Configure(param.cfg);
learner->Load(fi.get());
```

### 基于单样本的预测

+ 调用入口：`~/include/xgboost/learner.h`

```c++

```

+ 代码实现位置：

```c++
~/workspace/xgboost-exp/src/gbm/gbtree.cc
1. 函数 inline void PredLoopSpecalize
1.1. 调用inline float PredValue
// predict the leaf scores without dropped trees  inline float PredValue(const RowBatch::Inst &inst,                         int bst_group,                         unsigned root_index,                         RegTree::FVec *p_feats,                         unsigned tree_begin,                         unsigned tree_end) {    float psum = 0.0f;    p_feats->Fill(inst);    for (size_t i = tree_begin; i < tree_end; ++i) {      if (tree_info[i] == bst_group) {        bool drop = (std::binary_search(idx_drop.begin(), idx_drop.end(), i));        if (!drop) {          int tid = trees[i]->GetLeafIndex(*p_feats, root_index);          psum += weight_drop[i] * (*trees[i])[tid].leaf_value();        }      }    }    p_feats->Drop(inst);    return psum;  }
// 参数含义
// RowBatch::Inst &inst : 一条样本的数据结构
// bst_group : 二分类or多分类数，如果是二分类问题，该值为0；
// root_index : 表示每条样本起始root，在多任务学习中会用到。非多任务学习一般用不到默认为0，详细介绍在MetaInfo中；
// RegTree::FVec *p_feats : 回归树的数据结构
// tree_begin，tree_end分别表示模型中tree的个数起始和结束index。如果模型有300棵树，那么分别对应的值为0和299.
```

单样本调用时，可以按照如下格式配置参数：

```c++
PredValue(batch[i], 0, 0, &feats, tree_begin, tree_end);
```

