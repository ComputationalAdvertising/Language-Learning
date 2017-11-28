## 《大规模机器学习技术与OpenMIT》系列：参数服务器详解

目录

+ [写在前面](#0.写在前面)
+ [参数服务器架构](#1.参数服务器架构) 
+ [PS系统启动过程](#2.PS系统启动过程)
+ [PS任务启动过程](#3.PS任务启动过程)
    + [Scheduler节点启动过程](#3.1.Scheduler节点启动过程)
    + [Server节点启动过程](#3.2.Server节点启动过程)   
    + [Worker节点启动过程](#3.3.Worker节点启动过程)   
+ [PS节点通信与消息处理](#4.PS节点通信与消息处理)
    + [Worker与Server通信](#4.1.Worker与Server通信)
    + [Worker与Scheduler通信](#4.2.Worker与Scheduler通信)
    + [Server与Scheduler通信](#4.3.Server与Scheduler通信) 
+ [PS系统结束过程](#5.PS系统结束过程) 
+ [PS心跳管理与系统容错](#6.PS心跳管理与系统容错) 
+ [PS系统与Yarn环境交互](#7.PS系统与Yarn环境交互)

<h3 id="0.写在前面">0. 写在前面</h3>

<h3 id="1.参数服务器架构">1. 参数服务器架构</h3>

<h3 id="2.PS系统启动过程">2. PS系统启动过程</h3>

前面了解到，PS任务集群由1个scheduler节点、m个server节点和n个worker节点组成。这些节点中，首先启动的是scheduler节点，然后启动worker和server节点，这些节点的资源分配是由tracker中的PSTracker完成。这里我们说的PS启动过程包括以下几个核心问题：

2. scheduler节点如何管理worker／server节点的？
3. scheduler节点如何做心跳管理和容错的？

PS启动是从调用`PS::Start()`或`PS::StartAsync()`方法开始的，两个方法的区别在于前者启动是阻塞的，即启动完自身节点后会一直等待直至所有节点都启动完成；后者则不等待其它节点。对应的方法：

```c++
// This function will block until every nodes are started.
inline void Start(const char* argv0 = nullptr) {
  Postoffice::Get()->Start(argv0, true);
}
// This function will NOT block.
inline void StartAsync(const char* argv0 = nullptr) {
  Postoffice::Get()->Start(argv0, false);
}
```

从代码可以看到，它们都是调用了`Postoffice`类中的`Start`方法。我们看这个方法的实现过程。 `Postoffice`是一个单例类，`Postoffice::Get()`生成一个单例对象。我们先看下`Get`过程发生了什么？核心是`Postoffice`的构造函数：

```c++
Postoffice::Postoffice() {
  van_ = Van::Create("zmq");
  env_ref_ = Environment::_GetSharedRef();
  const char* val = NULL;
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_WORKER"));
  num_workers_ = atoi(val);
  val =  CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_SERVER"));
  num_servers_ = atoi(val);
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  std::string role(val);
  is_worker_ = role == "worker";
  is_server_ = role == "server";
  is_scheduler_ = role == "scheduler";
  verbose_ = GetEnv("PS_VERBOSE", 0);
}
```

邮局`Postoffice`构造函数比较简单，它首先创建自己的“货车”成员`van_`（用于传送接收消息用），然后获取环境变量，并从环境变量中知道了集群worker个数（`DMLC_NUM_WORKER`）、server个数（`DMLC_NUM_SERVER`）以及当前节点的角色（`DMLC_ROLE`，是"worker"或"server"或"scheduler"），最后初始化日志等级`verbose_`。

在`Start`方法中做了哪些事情呢？ 初始化log没什么好说的，在初始化节点信息方面以worker为例：

```c++
void Postoffice::Start(const char* argv0, const bool do_barrier) {
  // init glog
  if (argv0) {
    dmlc::InitLogging(argv0);
  } else {
    dmlc::InitLogging("ps-lite\0");
  }

  // init node info.
  for (int i = 0; i < num_workers_; ++i) {
    int id = WorkerRankToID(i);
    for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
            kWorkerGroup + kScheduler,
            kWorkerGroup + kServerGroup + kScheduler}) {
      node_ids_[g].push_back(id);
    }
  }
  // ...
  // do a barrier here
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
}
```

构造函数我们知道了worker的个数，那么这里每个worker节点的序号（rank）为`{0, .., num_worker-1}`。但是对于sever节点的序号来说也是从0开始的，从全局看rank＝0的节点分不清属于worker还是server。这里Postoffice对全局节点进行了统一的编号，对于worker和server节点分别根据`WorkerRankToID`和`ServerRankToID`方法进行节点编号转化。转化过程也比较简单：

```c++
  static inline int WorkerRankToID(int rank) {
    return rank * 2 + 9;
  }
  static inline int ServerRankToID(int rank) {
    return rank * 2 + 8;
  }
```

我们可以看到转化后的worker ID都是奇数（从9开始），server ID都是偶数（从8开始）。那么问题来了：为什么从8开始而不是从1开始编号？1到7的ID干嘛用了？ 这些ID由专门的用途，具体见下表：

| ID | 用途 | ID | 用途 |
| :--: | --- | :--: | --- |
| 1 | `kScheduler` | 3 | `kScheduler + kServerGroup` |
| 2 | `kServerGroup` | 5 | `kScheduler + kWorkerGroup` |
| 4 | `kWorkerGroup` | 6 | `kServerGroup + kWorkerGroup` |
| - | - | 7 | `kScheduler + kServerGroup + kWorkerGroup` |
    
我们可以看到，1到7的ID用于表示scheduler节点、组节点以及跨组的组合ID了。初始化节点信息主要是在**为不同的组ID对应的ID集合**，成员变量`std::unordered_map<int, std::vector<int>> node_ids_;`存储的就是每个ID对应的与之相关的ID集合。

注意：到这里为止，划分节点信息的归属，但是并没有将每个节点的信息（hostname/port等） 与ID“绑定”起来。怎么完成“绑定”过程？这时候需要启动“货车”`ZMQVan::Start`来完成。

zmq的初始化比较简单，new一个zmq对象，并设置最大的sockets。 

```c++
ZMQVan::Start() override {
  void * context_ = zqm_ctx_new();
  zmq_ctx_set(context_, MAX_SOCKETS, 65536);
  Van::Start();
}
```
“货车”`Van::Start()`启动会完成以下功能：

1. 获取scheduler信息和获取当前节点信息。主要是从环境变量中获取，包括hostname, port等，保存在`Node`对象中。
> 之所以每个节点都要获取scheduler信息，是因为都需要与scheduler主动通信，第一次通信获取该节点对应的ID。

2. 绑定端口，并与scheduler节点建立连接；
3. 开一个线程用于接收消息。`new thread(&Van::Receiving, this)`；
4. 向scheduler发送消息（scheduler除外）。主动向scheduler发送消息，让scheduler知道自己（当前节点）的存在.
5. 初始化消息重发功能（`resender_`）和建立（与scheduler的）心跳线程；

下面的代码与上述5个功能是对应着的：

```c++
void Van::Start() {
  // get scheduler info
  scheduler_.hostname = std::string(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_URI")));
  scheduler_.port     = atoi(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_PORT")));
  scheduler_.role     = Node::SCHEDULER;
  scheduler_.id       = kScheduler;
  is_scheduler_       = Postoffice::Get()->is_scheduler();

  // get my node info
  if (is_scheduler_) {
    my_node_ = scheduler_;
  } else {
    auto role = is_scheduler_ ? Node::SCHEDULER :
                (Postoffice::Get()->is_worker() ? Node::WORKER : Node::SERVER);
    const char* nhost = Environment::Get()->find("DMLC_NODE_HOST");
    // 省略，ip，role，port从环境变量获取部分 ...
    my_node_.hostname = ip;
    my_node_.role     = role;
    my_node_.port     = port;
    // cannot determine my id now, the scheduler will assign it later
    // set it explicitly to make re-register within a same process possible
    my_node_.id = Node::kEmpty;
  }

  // bind.
  my_node_.port = Bind(my_node_, is_scheduler_ ? 0 : 40);
  PS_VLOG(1) << "Bind to " << my_node_.DebugString();
  CHECK_NE(my_node_.port, -1) << "bind failed";

  // connect to the scheduler
  Connect(scheduler_);

  // for debug use
  if (Environment::Get()->find("PS_DROP_MSG")) {
    drop_rate_ = atoi(Environment::Get()->find("PS_DROP_MSG"));
  }
  // start receiver
  receiver_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Receiving, this));

  if (!is_scheduler_) {
    // let the scheduler know myself
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::ADD_NODE;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
  // wait until ready
  while (!ready_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // resender
  if (Environment::Get()->find("PS_RESEND") && atoi(Environment::Get()->find("PS_RESEND")) != 0) {
    int timeout = 1000;
    if (Environment::Get()->find("PS_RESEND_TIMEOUT")) {
      timeout = atoi(Environment::Get()->find("PS_RESEND_TIMEOUT"));
    }
    resender_ = new Resender(timeout, 10, this);
  }

  if (!is_scheduler_) {
    // start heartbeat thread
    heartbeat_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Heartbeat, this));
  }
}
```

这里要注意的是，第22行，这时候当前节点的ID还不知道，等着后续scheduler分配，这里先赋值`Node::kEmpty`；第41行~49行，表示当前节点第一次向scheduler主动通信的过程，把自身的节点信息放在msg里，发送出去。其中，消息接收者recver=kScheduler; 控制命令为**添加节点**control.cmd=Control::ADD_NODE；而scheduler收到控制命令为`Control::ADD_NODE`时，会为收到的节点分配ID。这部分后面会详细介绍。

至此，我们知道每个节点启动时要做的事情，现在回到"邮局"`Postoffice::Get()->Start`部分：

```c++
void Postoffice::Start(const char* argv0, const bool do_barrier) {
  // ...
  // start van
  van_->Start();

  // record start time
  start_time_ = time(NULL);

  // do a barrier here
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
}
```

最后一步`Barrier`方法是阻塞操作，直至所有的节点都完成启动。下面分析下这个方法：

```c++
void Postoffice::Barrier(int node_group) {
  if (GetNodeIDs(node_group).size() <= 1) return;
  // 省略check过程 .... 
  std::unique_lock<std::mutex> ulk(barrier_mu_);
  barrier_done_ = false;
  Message req;
  req.meta.recver = kScheduler;
  req.meta.request = true;
  req.meta.control.cmd = Control::BARRIER;
  req.meta.control.barrier_group = node_group;
  req.meta.timestamp = van_->GetTimestamp();
  CHECK_GT(van_->Send(req), 0);

  barrier_cond_.wait(ulk, [this] {
      return barrier_done_;
    });
}
```

我们可以看到，Barrier函数里，当前节点向scheduler请求`Control::BARRIER`命令，等待kscheduler直到`barrier_done_`为true。这个过程用`ulk`互斥量将`barrier_mu_`锁住。

scheduler收到各个节点命令为`Control::BARRIER`的请求时，它的处理过程如下(`src/van.cc Receiving方法`)：

```c++
      // Receiving其它功能 ...
      } else if (ctrl.cmd == Control::BARRIER) {
        if (msg.meta.request) {   // scheduler处理过程
          if (barrier_count_.empty()) {
            barrier_count_.resize(8, 0);
          }
          int group = ctrl.barrier_group;
          ++barrier_count_[group];
          PS_VLOG(1) << "Barrier count for " << group << " : " << barrier_count_[group];
          if (barrier_count_[group] ==
              static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())) {
            barrier_count_[group] = 0;
            Message res;
            res.meta.request = false;
            res.meta.control.cmd = Control::BARRIER;
            for (int r : Postoffice::Get()->GetNodeIDs(group)) {
              res.meta.recver = r;
              res.meta.timestamp = timestamp_++;
              CHECK_GT(Send(res), 0);
            }
          }
        } else {  // 当前节点收到scheduler后的处理
          Postoffice::Get()->Manage(msg);
        }
```

`barrier_count_`是一个size为8的数组，存储每个group对应的node数量，scheduler每收到一个`Control::BARRIER`请求，都会在相应的group下标上累加`++barrier_count_[group]`。

当收到的请求为group的数量等于全局初始化时每个group对应的节点数相同时（即所有的group相关的节点`Control::BARRIER`请求均已收到）：`barrier_count_[group]=static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())`，这时scheduler将`barrier_count_`归0，并向所有的请求节点发送`Control::BARRIER`命令收到信号`CHECK_GT(Send(res), 0)`。所谓的Barrier阻塞直至所有节点完成启动，核心在于所有计算节点都向scheduler发送`BARRIER`信号，由scheduler统一控制**节奏**。

对于当前节点来说，收到scheduler发来的`Control::BARRIER`命令后（对应的`request=false`），就知道其它所有节点也完成启动了，此时会执行自己的Manage操作（调用`Postoffice::Get()->Manage(msg)`）。我们看Manage的处理逻辑：

```c++
void Postoffice::Manage(const Message& recv) {
  CHECK(!recv.meta.control.empty());
  const auto& ctrl = recv.meta.control;
  if (ctrl.cmd == Control::BARRIER && !recv.meta.request) {
    barrier_mu_.lock();
    barrier_done_ = true;
    barrier_mu_.unlock();
    barrier_cond_.notify_all();
  }
}
```

Manage处理逻辑比较简单，就是对变量`barrier_done`置为true，唤醒其它等待该变量（对象锁）的线程，在启动阶段对应`Postoffice::Barrier`里面的`barrier_cond_.wait(ulk, [this] {return barrier_done_; })`。这一步对于所有的非scheduler节点处理逻辑都一样。

到这里为止，PS整体初始化过程就已经完成了，这里简要概括一下主要工作过程：

+ 首先，自身节点的初始化，包括初始化通信模块(Van)，获取scheduler信息等；
+ 其次，与scheduler节点建立连接，获取全局节点ID；
+ 其次，当前节点与scheduler交互（交互命令：`Control::BARRIER`），同步完成全局节点初始化（由scheduler控制节奏）。

<h3 id="3.PS任务启动过程">3. PS任务启动过程</h3>

PS机器学习任务计算过程需要scheduler、worker、server三方节点参与，这三方的计算任务均要启动。我们先看scheduler节点的启动过程。 

<h4 id="3.1.Scheduler节点启动过程">3.1. Scheduler节点启动过程</h4>

scheduler启动过程比较简单，首先创建一个`SimpleApp`对象，用于App层面的请求响应处理。具体的请求响应处理逻辑由对应的计算任务决定，与`SimpleApp`无关，开发者只需要按照约定接口实现自己的请求响应逻辑，然后注册给`SimpleApp`即可。

请求响应约定接口：`using Handle = std::function<void(const SimpleData& recved, SimpleApp* app)>;`

假如开发者实现的请求响应逻辑如下（仅用于示例，与具体项目无关）：

```c++
void Scheduler::Handle(const ps::SimpleData & recved, ps::SimpleApp * app) {
  ps::Message msg;
  msg.meta.head = recved.head;
  msg.meta.body = recved.body;
  msg.meta.timestamp = recved.timestamp;
  msg.meta.request = false;
  msg.meta.simple_app = true;
  msg.meta.customer_id = scheduler_->get_customer()->id();
  msg.meta.recver = recved.sender;
  msg.meta.sender = ps::Postoffice::Get()->van()->my_node().id;

  int cmd = recved.head;
  switch(cmd) {
    case signal::METRIC:
      { // 处理逻辑 ... }
      break;
    case signal::WORKER_FINISH:
      { // 处理逻辑 ... }
      break;
    case signal::SERVER_FINISH:
      { // 处理逻辑 ... }
      break;
    default:
      LOG(FATAL) << "can not recognize cmd.";
  }
}
```

创建一个`SimpleApp`对象，并将自定义的请求响应逻辑注册到`SimpleApp`中，示例如下：

```c++
  scheduler_app_.reset(new ps::SimpleApp(0));
  // register request processing func
  using namespace std::placeholders;
  scheduler_app_->set_request_handle(std::bind(&Scheduler::Handle, this, _1, _2));
  scheduler_app_->set_response_handle(std::bind(&Scheduler::Handle, this, _1, _2));
```

上述代码可以放在Scheduler任务的构造函数中，在启动Scheduler任务时，只需要创建一下Scheduler任务即可。如何保证Scheduler任务在整个计算过程常驻并能在任务结束时退出？这是可以借助互斥量和条件变量来控制。

我们重点关注创建一个`SimpleApp`对象的过程中，需要做哪些事情？

首先调用`SimpleApp`一个默认的构造函数，用于初始化请求和响应变量的默认执行逻辑。

```c++
  inline SimpleApp() : obj_(nullptr) {
    request_handle_ = [](const SimpleData& recved, SimpleApp* app) {
      app->Response(recved);
    };
    response_handle_ = [](const SimpleData& recved, SimpleApp* app) { };
  }
```

`obj_`变量是`SimpleApp`的成员变量，需要用“顾客”类－`Customer`－来创建，创建时需要绑定`SimpleApp`整体的处理逻辑。

```c++
inline SimpleApp::SimpleApp(int app_id) : SimpleApp() {
  using namespace std::placeholders;
  obj_ = new Customer(app_id, std::bind(&SimpleApp::Process, this, _1));
}
```

其中`SimpleApp::Process`逻辑是固定的，仅需要根据`msg.meta.request`字段判断是请求还是响应，然后执行开发者自定义的处理逻辑（如果开发者没有注册自定义处理逻辑，会调用默认逻辑）。

```c++
inline void SimpleApp::Process(const Message& msg) {
  SimpleData recv;
  recv.sender    = msg.meta.sender;
  recv.head      = msg.meta.head;
  recv.body      = msg.meta.body;
  recv.timestamp = msg.meta.timestamp;
  if (msg.meta.request) {
    CHECK(request_handle_);
    request_handle_(recv, this);
  } else {
    CHECK(response_handle_);
    response_handle_(recv, this);
  }
}
```

此外，`SimpleApp`中还提供了`Request`和`Response`接口，供开发者用于节点之间的消息请求和响应反馈。接口格式为：

```c++
inline int SimpleApp::Request(int req_head, const std::string& req_body, int recv_id) {
  // setup message
  Message msg;
  // 省略封装message过程 ...
  // send
  for (int r : Postoffice::Get()->GetNodeIDs(recv_id)) {
    msg.meta.recver = r;
    Postoffice::Get()->van()->Send(msg);
  }
  return ts;
}

inline void SimpleApp::Response(const SimpleData& req, const std::string& res_body) {
  // setup message
  // 省略封装message过程 ....
  // send
  Postoffice::Get()->van()->Send(msg);
}
```

下面我们重点分析创建并绑定了`SimpleApp::Process`的顾客`Customer`对象`obj_`干什么用的？  

我们先看`obj_`在`SimpleApp`类中都做了什么？

+ `int ts = obj_->NewRequest(recv_id);`：为新请求生成一个时间戳；
+ `msg.meta.customer_id = obj_->id();`：在Request／Response中填充`customer_id`信息；
+ `obj_->WaitRequest(timestamp);`：同步请求操作，以ts为key；

一句话概括`Customer`类的功能：**针对每一次请求／响应进行控制**， 包括WaitRequest，NewRequest，NumResponse，Receiving等。Receiving方法的功能是等待取出消息队列`recv_queue_`中的消息供`Customer`中的`recv_handle_`“消费”，而`recv_handle_`是在创建`SimpleApp`或`KVWorker`或`KVServer`时传入`Customer`构造函数中的。

尤其要说明的是：消息队列`recv_queue_`中的消息是由货车`Van`通过调用`Customer::Accept`函数“放到”队列中的。

```c++
void Van::Receiving() {
  const char* heartbeat_timeout_val = Environment::Get()->find("PS_HEARTBEAT_TIMEOUT");
  const int heartbeat_timeout = heartbeat_timeout_val ? atoi(heartbeat_timeout_val) : kDefaultHeartbeatInterval;
  Meta nodes;  // for scheduler usage
  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);
    // 省略 ....
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;

    if (!msg.meta.control.empty()) {  // Control信号
      // do some management
      auto& ctrl = msg.meta.control;
      // 省略针对控制信号逻辑的处理 ...
    } else {   // 非Control信息，即数据信号
      CHECK_NE(msg.meta.sender, Meta::kEmpty);
      CHECK_NE(msg.meta.recver, Meta::kEmpty);
      CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
      int id = msg.meta.customer_id;
      auto* obj = Postoffice::Get()->GetCustomer(id, 5);
      CHECK(obj) << "timeout (5 sec) to wait App " << id << " ready";
      obj->Accept(msg);
    }
```

至此，我们搞清了scheduler任务节点的工作。这里在梳理下思路：

+ step1: 创建一个`SimpleApp`，过程包括用`SimpleApp::Process`方法创建一个`Customer`对象，`Process`会调用开发者自定义的请求／响应处理逻辑；
+ step2: 当向其它节点发送等待请求时，`Customer::Receiving`方法会循环取出消息队列中的消息（message），该消息可能是Request类型或者Response类型，最后都会交由`recv_handler_`来处理（处理逻辑由用户注册）。
+ step3: 消息队列中的消息是由`Van::Receiving`收到后直接给到指定顾客手里（调用`Customer::Accept`方法），由顾客调用用户自定义的Handle来进行进一步“消费”。
+ step4: 消息处理完后，要向对方发送响应。该响应要么是ack消息（应答），要么是对方请求的数据消息。

简单描述消息传输过程：

$$
\text{Van::Receiving} \xrightarrow{\text{msg}} \text{Customer::Accept} \xrightarrow{\text{msg}} \text{recv_queue_.WaitAndPop} \xrightarrow{\text{msg}} \text{recv_handle_} \xrightarrow{\text{(updated) msg}} \text{Response/Send} 
$$

<h4 id="3.2.Server节点启动过程">3.2. Server节点启动过程</h4>

Server节点启动过程与Scheduler节点启动非常类似，最大的区别在于后者用`SimpleApp`来创建对象，Server节点启动用的是`KVServer`来创建对象。从`KVServer`的实现可以看到，`KVServer`继承了`SimpleApp`，在子类中基础上添加Server节点的逻辑，主要是与Worker节点信息交互方面的处理逻辑（`pull/push`操作）。

```c++
template <typename Val>
class KVServer : public SimpleApp {
 public:
  explicit KVServer(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, std::bind(&KVServer<Val>::Process, this, _1));
  }
  
  virtual ~KVServer() { delete obj_; obj_ = nullptr; }
  
  using ReqHandle = std::function<void(const KVMeta& req_meta,
                                       const KVPairs<Val>& req_data,
                                       KVServer* server)>;
  void set_request_handle(const ReqHandle& request_handle) {
    CHECK(request_handle) << "invalid request handle";
    request_handle_ = request_handle;
  }

  void Response(const KVMeta& req, const KVPairs<Val>& res = KVPairs<Val>());

 private:
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief request handle */
  ReqHandle request_handle_;
};
```

我们可以看到，`KVServer`与`SimpleApp`很像，构造函数中都会创建一个`Customer`类，用于处理每一次的`pull`/`push`请求。`Server<Val>::Process`方法中的`request_handle_`仍由用户自定义并注册给`KVServer`。

> 在`SimpleApp`中，`Customer`用于处理`request`/`response`请求。

`KVServer`中的`Customer`对象工作过程在Scheduler节点启动过程中已经详细介绍，这里不在赘述。

值得注意的是，`KVServer`在整个任务中消息处理主要分两种：

1. `simple_app`类型消息，多是控制命令消息，用于系统中节点之间的协同工作，对应`Request/Response`操作；
2. 非`simple_app`类型消息，多是Server与Worker之间的数据通信，数据封装在`KVPairs`中，对应`Pull/Push`操作；

<h4 id="3.3.Worker节点启动过程">3.3. Worker节点启动过程</h4>

Worker节点的启动与Server的区别在于前者是调用`KVWorker`来完成初始化。`KVWorker`相比`SimpleApp`和`KVServer`，主要多了两个关键操作：`Pull`和`Push`.

```c++
template<typename Val>
class KVWorker : public SimpleApp {
 public:
  using SimpleApp::obj_;
  
  explicit KVWorker(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    slicer_ = std::bind(&KVWorker<Val>::DefaultSlicer, this, _1, _2, _3);
    obj_ = new Customer(app_id, std::bind(&KVWorker<Val>::Process, this, _1));
  }

  int Push(const std::vector<Key>& keys, const std::vector<Val>& vals, const std::vector<int>& lens = {},
           int cmd = 0, const Callback& cb = nullptr) {
    return ZPush(SArray<Key>(keys), SArray<Val>(vals), SArray<int>(lens), cmd, cb);
  }

 int Pull(const std::vector<Key>& keys, std::vector<Val>* vals, std::vector<int>* lens = nullptr,
           int cmd = 0, const Callback& cb = nullptr) {
    return Pull_(SArray<Key>(keys), vals, lens, cmd, cb);
  }
  // 省略 ...
```

Worker节点启动过程与Scheduler、Server一样。唯一的区别在于，Worker启动时不需要自定义请求处理逻辑。**对于PS机器学习任务来讲，Worker端的行为都是主动行为，比如数据通信则主动向Server端执行Pull/Push请求，（控制）消息传输向Server端或Scheduler执行Request/Response请求。** 

至此，我们把PS机器学习任务中的节点启动过程以及内部的消息传递过程介绍完毕。下面我们重点关注任务启动之后的计算过程，节点之间是如何“对话”的？

<h3 id="4.PS节点通信与消息处理">4. PS节点通信与消息处理</h3>

PS机器学习任务启动之后，计算过程中节点之间是如何协同工作的？协同工作主要依靠通信来完成。不同节点类型承载着不同的任务，分布式系统和计算逻辑要把相应的消息发送给相应的节点去执行。

下面我们分别看Worker、Server、Scheduler两两之间的具体通信过程。

<h4 id="4.1.Worker与Server通信">4.1. Worker与Server通信</h4>

<h4 id="4.2.Worker与Scheduler通信">4.2. Worker与Scheduler通信</h4>

<h4 id="4.3.Server与Scheduler通信">4.3. Server与Scheduler通信</h4>

<h3 id="5.PS系统结束过程">5. PS系统结束过程</h3>

上面我们详细介绍了参数服务器系统启动、节点任务启动以及计算过程中如何通信和处理消息的。当PS完成整个计算过程时，每个节点是如何结束并退出的呢？

我们知道在节点任务启动时，里面用`ps::RegisterExitCallback()`方法来注册任务退出时用户自定义的方法（退出时的回调函数）。各个节点的退出回调函数注册如下（示例）：

```c++
ps::RegisterExitCallback([scheduler]() { delete scheduler; });
ps::RegisterExitCallback([server]() { delete server; });
ps::RegisterExitCallback([worker]() { delete worker; });
```

在调用回调函数之前，PS还做了一些事情，我们还是从`PS::Finalize()`开始。`PS::Finalize()`调用了`Postoffice::Get()->Finalize(do_barrier)`，参数`do_barrier`指的是**是否需要等待直至所有节点都结束（阻塞过程）。**

```c++
void Postoffice::Finalize(const bool do_barrier) {
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
  van_->Stop();
  if (exit_callback_) exit_callback_();
}
```

如果`do_barrier`为true，执行阻塞函数`Barrier`，等待全部节点完成计算，然后进入终止“货车”的功能。参数的含义是覆盖全部计算节点。该过程与`PS::Start`中的逻辑是一样的。

终止“货车”的过程如下：

```c++
// ZMQVan::Stop
  void Stop() override {
    PS_VLOG(1) << my_node_.ShortDebugString() << " is stopping";
    Van::Stop();
    // close sockets
    int linger = 0;
    int rc = zmq_setsockopt(receiver_, ZMQ_LINGER, &linger, sizeof(linger));
    CHECK(rc == 0 || errno == ETERM);
    CHECK_EQ(zmq_close(receiver_), 0);
    for (auto& it : senders_) {
      int rc = zmq_setsockopt(it.second, ZMQ_LINGER, &linger, sizeof(linger));
      CHECK(rc == 0 || errno == ETERM);
      CHECK_EQ(zmq_close(it.second), 0);
    }
    zmq_ctx_destroy(context_);
  }
  
void Van::Stop() {
  // stop threads
  Message exit;
  exit.meta.control.cmd = Control::TERMINATE;
  exit.meta.recver = my_node_.id;
  SendMsg(exit);
  receiver_thread_->join();
  if (!is_scheduler_) heartbeat_thread_->join();
  if (resender_) delete resender_;
}
```

由于van默认由`zmq`初始化的，它首先调用父类的Stop方法，结束任务层面对象，然后结束自己的通信层面对象。父类Stop方法首先向自己发送命令为`Control::TERMINATE`的消息，目的是从`Van::Receiving`消息接收函数中的`while(true)`跳出来。然后释放接收线程，如果当前节点是scheduler节点的话，还要释放心跳线程`heartbeat_thread_`。Van中还有一个`resender`对象（消息重发和监控告警用），此时也要释放掉。

`Postoffice::Finalize`方法的最后一步是执行用户自定义的退出回调函数`exit_callback_()`。

至此，PS系统的结束过程介绍完了。 

<h3 id="6.PS心跳管理与系统容错">6. PS心跳管理与系统容错</h3>

<h3 id="7.PS系统与Yarn环境交互">7. PS系统与Yarn环境交互</h3>
 



