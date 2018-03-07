## RDMA-Verbs编程

+ author: zhouyongsdzh@foxmail.com


### 目录

+ RDMA介绍
+ RDMA-Verbs编程

## RDMA-Verbs编程

verbs是RDMA的CoEC解决方案的一种编程方式。为了让应用程序在RDMA网络上执行，需要用verbs开发RDMA通信程序。verbs编程与TCP／IP编程逻辑类似，最大的区别在于实现方式不同，以及verbs程序逻辑更加复杂。

在介绍verbs编程之前，我们先看下tcp／ip协议下机器之间是如何通信的？示例：

#### TCP/IP协议RPC通信程序工作过程

Server端：**创建套接字（`socket`）$\longrightarrow$ 绑定端口（`bind`）$\longrightarrow$ 监听连接请求（`listen`）$\longrightarrow$ 等待请求链接（`accept`）**

Client端：**创建client端套接字（`socket`） $\longrightarrow$ 连接server socket（`connect`） $\longrightarrow$ 接收server端消息（`recv`） $\longrightarrow$ 向server端发送消息，&& 接收（`send/recv`）**

> 通信消息（message）放在`char buf[BUFFSIZE]`结构里。

verbs通信程序工作与上述类似，只是实现不同。verbs第一步是初始化所有资源`resource`。

### 资源初始化 

我们先把RDMA网络配置信息封装在一个结构体中，如下所示：

```c
/*!
 * \brief structure of test parameters
 * \param dev_name IB device name
 * \param server_name server host name
 * \param tcp_port server TCP port
 * \param ib_port local IB port to work with
 * \param gid_idx gid index to use
 */
struct config_t {
  const char* dev_name;
  char*       server_name;
  uint32_t    tcp_port;
  int         ib_port;
  int         gid_idx;
};
struct config_t config = {NULL, NULL, 19875, 1, -1};
```

解释下上述字段的含义，`dev_name`：设备名，即RDMA网络硬件名称（`ibv_devinfo`命令可以查看，如`hca_id:	mlx5_0`，设备名为`mlx5_0`）。`server_name`即server端机器的IP地址，在`connect`过程中会应用。`tcp_port`：TCP网络端口号。`ib_port`为rdma设备网络端口号（该字段信息不能像`tcp_port`一样随意指定，需要通过命令`ibv_devinfo`查看RDMA设备绑定了哪些端口，如`port:	1`说明绑定了1端口）；`gid_idx`子段也与设备有关，需要通过`show_gids`命令查看。

声明资源`struct resources res;`。这里的“资源”也是一个结构体： 

```c
/*!
 * \brief structure of system resources
 */
struct resources {
  struct ibv_device_attr    device_attr;    // device attributes
  struct ibv_port_attr      port_attr;      // IB port attributes
  struct cm_con_data_t      remote_props;   // values to connect to remote side
  struct ibv_context        *ib_ctx;        // device handle
  struct ibv_pd             *pd;            // PD handle
  struct ibv_cq             *cq;            // CQ handle
  struct ibv_qp             *qp;            // QP handle
  struct ibv_mr             *mr;            // MR handle for buf
  char                      *buf;           // memory buffer pointer, used for RDMA and send ops
  int                       sock;           // TCP socket file descriptor
}; // struct resources
```

我们可以看到，结构体`resources`中的成员大部分都是`ibv_`开头的结构体成员（是verbs的api），这里`struct cm_con_data_t`表示远程通信节点信息，定义如下： 

```c
/*!
 * \brief structure to exchange data which is needed to connect the QPs
 * \param addr buffer address
 * \param rkey remote key
 * \param qp_num QP number
 * \param lid LID of the IB port
 * \param gid[16]
 */
struct cm_con_data_t {
  uint64_t   addr;
  uint32_t   rkey;
  uint32_t   qp_num;
  uint16_t   lid;
  uint8_t    gid[16];
}__attribute__((packed));
```
声明完`resources`后，紧接着对其初始化：

```c
static void resources_init(struct resources* res) {
  memset(res, 0, sizeof * res);
  res->sock = -1;
}
```
可以看到，resources初始化非常简单，什么都没做。 下面才是真正的创建resources的流程。由于代码比较长，这里分开解读： 

+ **`resources`创建第1步：建立sock连接**

```c
/*!
 * \brief function resources create
 */
static int resources_create(struct resources* res) {
  struct ibv_device**       dev_list = NULL;
  struct ibv_qp_init_attr   qp_init_attr;
  struct ibv_device*        ib_dev = NULL;

  size_t                    size;
  int                       i;
  int                       mr_flags = 0;
  int                       cq_size = 0;
  int                       num_devices;
  int                       rc = 0;

  // if client side
  if (config.server_name) {
    res->sock = sock_connect(config.server_name, config.tcp_port);
    if (res->sock < 0) {
      fprintf(stderr, "failed to establish TCP connection to server %s, port %d\n",
             config.server_name, config.tcp_port);
      rc = -1;
      goto resources_create_exit;
    }
  } else {
    fprintf(stdout, "waiting on port %d for TCP connection\n", config.tcp_port);
    res->sock = sock_connect(NULL, config.tcp_port);
    if (res->sock < 0) {
      fprintf(stderr, "failed to establish TCP connection with client on port %d\n",
             config.tcp_port);
      rc = -1;
      goto resources_create_exit;
    }
  }
  fprintf(stdout, "TCP connection was established\n");
  fprintf(stdout, "searching for IB devices in host\n");
```

可以看到，建立sock连接的调用的是`sock_connect`函数，返回一个socket。它的具体实现如下：

```c
/*!
 * \brief function connect a socket.
 */
static int sock_connect(const char *servername, int port) {
  struct addrinfo     *resolved_addr = NULL;   // getaddrinfo返回后的（连接的）地址信息
  struct addrinfo     *iterator;
  char                service[6];
  int                 sockfd = -1;
  int                 listenfd = 0;
  int                 tmp;
  
  // ai_family = AF_INET 
  struct addrinfo hints = {
    .ai_flags   = AI_PASSIVE,
    .ai_family  = AF_INET,  
    .ai_socktype = SOCK_STREAM
  };

  if (sprintf(service, "%d", port) < 0) {
    goto sock_connect_exit;
  }

  // resolve DNS address, use sockfd as temp storage
  sockfd = getaddrinfo(servername, service, &hints, &resolved_addr);

  if (sockfd < 0) {
    fprintf(stderr, "%s for %s: %d\n", gai_strerror(sockfd), servername, port);
    goto sock_connect_exit;
  }

  // search through result and find the one we want
  for (iterator = resolved_addr; iterator; iterator = iterator->ai_next) {
    sockfd = socket(iterator->ai_family, iterator->ai_socktype, iterator->ai_protocol);
    if (sockfd >= 0) {
      if (servername) {
        // client mode. initiate connection to remote
        if ((tmp = connect(sockfd, iterator->ai_addr, iterator->ai_addrlen))) {
          fprintf(stdout, "failed connect\n");
          close(sockfd);
          sockfd = -1;
        }
      } else {
        // server mode. set up listening socket an accept a connection
        listenfd = sockfd;
        sockfd = -1;
        if (bind(listenfd, iterator->ai_addr, iterator->ai_addrlen)) {
          goto sock_connect_exit;
        }
        listen(listenfd, 1);
        sockfd = accept(listenfd, NULL, 0);
      }
    }
  } // end for

sock_connect_exit:
  if (listenfd) close(listenfd);
  if (resolved_addr) freeaddrinfo(resolved_addr);
  if (sockfd < 0) {
    if (servername) {
      fprintf(stderr, "Couldn't connect to %s: %d\n", servername, port);
    } else {
      perror("server accept");
      fprintf(stderr, "accept() failed\n");
    }
  }

  return sockfd;
} // function sock_connect
```

+ **`resources`创建第2步：获取本地IB设备信息&&指定工作设备**

```c
  // get device names in the system
  dev_list = ibv_get_device_list(&num_devices);
  if (!dev_list) {
    fprintf(stderr, "failed to get IB devices list\n");
    rc = 1;
    goto resources_create_exit;
  }
  // if there isn't any IB device in host
  if (! num_devices) {
    fprintf(stderr, "error found %d device(s)\n", num_devices);
    rc = 1;
    goto resources_create_exit;
  }
  fprintf(stdout, "found %d device(s)\n", num_devices);
  
  // search for the specific device we want to work with
  for (i = 0; i < num_devices; ++i) {
    if (!config.dev_name) {
      config.dev_name = strdup(ibv_get_device_name(dev_list[i]));
      fprintf(stdout, "device not specified, using first one found: %s\n", config.dev_name);
    }
		fprintf(stdout, "found dev_name[%d]: %s\n", i, ibv_get_device_name(dev_list[i]));
    if (!strcmp(ibv_get_device_name(dev_list[i]), config.dev_name)) {
      ib_dev = dev_list[i];
      break;
    }
  }
  // if the device wasn't found in host
  if (!ib_dev) {
    fprintf(stderr, "IB device %s wasn't found\n", config.dev_name);
    rc = 1;
    goto resources_create_exit;
  }
```

获取设备信息并check指定的设备。首先，调用`ibv_get_device_list(&num_devices)`函数获取所有IB设备信息。参数`num_devices`保存获取到的IB设备数。返回值`dev_list`的类型为`struct ibv_device**`。然后，初始化参与工作的IB设备，如果参数`config.dev_name`未指定，那么会选择`dev_list`中第一个设IB设备为工作设备(用`ib_dev`保存)，否则验证`config.dev_name`的合法性。

+ **`resources`创建第3步：打开设备，初始化port、PD、CQ、MR、内存、QP等**

```c
  // get device handle
  res->ib_ctx = ibv_open_device(ib_dev);
  if (!res->ib_ctx) {
    fprintf(stderr, "failed to open device %s\n", config.dev_name);
    rc = 1;
    goto resources_create_exit;
  }

  // we are now done with device list, free it
  ibv_free_device_list(dev_list);
  dev_list = NULL;
  ib_dev = NULL;

  // query port properties
  if (ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr)) {
    fprintf(stderr, "ibv_query_port %d failed\n", config.ib_port);
    rc = 1;
    goto resources_create_exit;
  }

  // allocate Protection Domain
  res->pd = ibv_alloc_pd(res->ib_ctx);
  if (!res->pd) {
    fprintf(stderr, "ibv_alloc_pd failed\n");
    rc = 1;
    goto resources_create_exit;
  }

  // each side will send only one WR, so Completion Queue with 1 entry is enough
  cq_size = 1;
  res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);
  if (!res->cq) {
    fprintf(stderr, "failed to create CQ with %u entries\n", cq_size);
    rc = 1;
    goto resources_create_exit;
  }

  // allocate the memory buffer that will hold the data
  size = MSG_SIZE;
  res->buf = (char *) malloc(size);
  if (!res->buf) {
    fprintf(stderr, "failed to malloc %u bytes to memory buffer\n", (uint32_t)size);
    rc = 1;
    goto resources_create_exit;
  }
  memset(res->buf, 0, size);

  // only in the server side put the message in the memory buffer
  if (!config.server_name) {
    strcpy(res->buf, MSG);
    fprintf(stdout, "going to send the message: '%s'\n", res->buf);
  } else {
    memset(res->buf, 0, size);
  }

  // register the memory buffer
  mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);
  if (!res->mr) {
    fprintf(stderr, "ibv_get_mr failed with mr_flags = 0x%x\n", mr_flags);
    rc = 1;
    goto resources_create_exit;
  }

  fprintf(stdout, "MR was registered with addr=%p,lkey=0x%x,rkey=0x%x,flags=0x%x\n",
          res->buf, res->mr->lkey, res->mr->rkey, mr_flags);

  // create the Queue Pair
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.qp_type              = IBV_QPT_RC;
  qp_init_attr.sq_sig_all           = 1;
  qp_init_attr.send_cq              = res->cq;
  qp_init_attr.recv_cq              = res->cq;
  qp_init_attr.cap.max_send_wr      = 1;
  qp_init_attr.cap.max_recv_wr      = 1;
  qp_init_attr.cap.max_send_sge     = 1;
  qp_init_attr.cap.max_recv_sge     = 1;

  res->qp = ibv_create_qp(res->pd, &qp_init_attr);
  if (!res->qp) {
    fprintf(stderr, "failed to create QP\n");
    rc = 1;
    goto resources_create_exit;
  }
  fprintf(stdout, "QP was created, QP number=0x%x\n", res->qp->qp_num);

resources_create_exit:
  if (rc) {
    // error encountered, cleanup
    if (res->qp) { ibv_destroy_qp(res->qp); res->qp = NULL; }
    if (res->mr) { ibv_dereg_mr(res->mr); res->mr = NULL; }
    if (res->buf) { free(res->buf); res->buf = NULL; }
    if (res->cq) { ibv_destroy_cq(res->cq); res->cq = NULL; }
    if (res->pd) { ibv_dealloc_pd(res->pd); res->pd = NULL; }
    if (res->ib_ctx) { ibv_close_device(res->ib_ctx); res->ib_ctx = NULL; }
    if (dev_list) { ibv_free_device_list(dev_list); dev_list = NULL; }
    if (res->sock >= 0) {
      if (close(res->sock)) fprintf(stderr, "failed to close socket\n");
      res->sock = -1;
    }
  }
  return rc;
}
```

说明下部分重要的IB函数。

+ `res->ib_ctx = ibv_open_device(ib_dev)`：打开指定的IB设备，返回ib context，用于后续通信操作；
+ `ibv_free_device_list(dev_list);`：释放设备信息；
+ `ibv_query_port(res->ib_ctx, config.ib_port, &res->port_attr)`：初始化`res->port_attr`端口参数；
+ `res->pd = ibv_alloc_pd(res->ib_ctx)`：分配保护区间。何为PD？
+ `res->cq = ibv_create_cq(res->ib_ctx, cq_size, NULL, NULL, 0);`：创建完成队列?
+ ` res->mr = ibv_reg_mr(res->pd, res->buf, size, mr_flags);` 没搞明白什么意思？
+ `res->qp = ibv_create_qp(res->pd, &qp_init_attr);`：创建队列pair ？

**QP初始化与创建**

```c
struct ibv_qp_init_attr   qp_init_attr;
// create the Queue Pair
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));

  qp_init_attr.qp_type              = IBV_QPT_RC;
  qp_init_attr.sq_sig_all           = 1;
  qp_init_attr.send_cq              = res->cq;
  qp_init_attr.recv_cq              = res->cq;
  qp_init_attr.cap.max_send_wr      = 1;
  qp_init_attr.cap.max_recv_wr      = 1;
  qp_init_attr.cap.max_send_sge     = 1;
  qp_init_attr.cap.max_recv_sge     = 1;

  res->qp = ibv_create_qp(res->pd, &qp_init_attr);
```

可以看到，QP的类型为`IBV_QPT_RC`。`qp_init_attr.cap`的数据类型为：

```c++
struct ibv_qp_cap {
  uint32_t max_send_wr;      // Maximum number of outstanding send requests in the send queue 
  uint32_t max_recv_wr;      // Maximum number of outstanding recv requests in the recv queue 
  uint32_t max_send_sge;    // Maximum number of outstanding scatter/gather elements (SGE) in the send queue 
  uint32_t max_recv_sge;
  uint32_t max_inline_data;   // Maximum size in bytes of inline data on the send queue
};
```

### 连接Queue Pair

创建完resources后，接下来是用resources连接Queue Pair. 实现过程如下：

```c
/*!
 * \brief function connect_qp
 */
static int connect_qp(struct resources* res) {
  struct cm_con_data_t    local_con_data;
  struct cm_con_data_t    remote_con_data;
  struct cm_con_data_t    tmp_con_data;
  int                     rc = 0;
  char                    temp_char;
  union ibv_gid           my_gid;

  if (config.gid_idx >= 0) {
    rc = ibv_query_gid(res->ib_ctx, config.ib_port, config.gid_idx, &my_gid);
    if (rc) {
      fprintf(stderr, "could not get gid for port %d, index %d\n", config.ib_port, config.gid_idx);
      return rc;
    }
  } else {
    memset(&my_gid, 0, sizeof(my_gid));
  }

  // exchange using TCP sockets info required to connect QPs
  local_con_data.addr     = htonll((uintptr_t) res->buf);
  local_con_data.rkey     = htonl(res->mr->rkey);
  local_con_data.qp_num   = htonl(res->qp->qp_num);
  local_con_data.lid      = htons(res->port_attr.lid);
  memcpy(local_con_data.gid, &my_gid, 16);

  fprintf(stdout, "\nLocal LID = 0x%x\n", res->port_attr.lid);
  if (sock_sync_data(res->sock, sizeof(struct cm_con_data_t), (char*)&local_con_data, (char*)&tmp_con_data) < 0) {
    fprintf(stderr, "failed to exchange connection data between sides\n");
    rc = 1;
    goto connect_qp_exit;
  }

  remote_con_data.addr      = ntohll(tmp_con_data.addr);
  remote_con_data.rkey      = ntohl(tmp_con_data.rkey);
  remote_con_data.qp_num    = ntohl(tmp_con_data.qp_num);
  remote_con_data.lid       = ntohs(tmp_con_data.lid);
  memcpy(remote_con_data.gid, tmp_con_data.gid, 16);

  // save the remote side attributes, we will need it for the post SR
  res->remote_props = remote_con_data;

  fprintf(stdout, "Remote address = 0x%ld\n", remote_con_data.addr);
  fprintf(stdout, "Remote rkey = 0x%x\n", remote_con_data.rkey);

  fprintf(stdout, "Remote QP number = 0x%x\n", remote_con_data.qp_num);
  fprintf(stdout, "Remote LID = 0x%x\n", remote_con_data.lid);
  if (config.gid_idx >= 0) {
    uint8_t* p = remote_con_data.gid;
    fprintf(stdout, "Remote GID = %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n",
           p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],p[15]);
  }
  // modify the QP to init
  rc = modify_qp_to_init(res->qp);
  if (rc) {
    fprintf(stderr, "change QP state to INIT failed\n");
    goto connect_qp_exit;
  }

  // let the client post RR to be prepared for incoming messages
  if (config.server_name) {
    rc = post_receive(res);
    if (rc) {
      fprintf(stderr, "failed to post RR\n");
      goto connect_qp_exit;
    }
  }

  // modify the QP to RTR
  rc = modify_qp_to_rtr(res->qp, remote_con_data.qp_num, remote_con_data.lid, remote_con_data.gid);
  if (rc) {
    fprintf(stderr, "failed to modify QP state to RTR\n");
    goto connect_qp_exit;
  }

  rc = modify_qp_to_rts(res->qp);
  if (rc) {
    fprintf(stderr, "failed to modify QP state to RTR\n");
    goto connect_qp_exit;
  }

  fprintf(stdout, "QP state was change to RTS\n");

  // sync to make sure that boolth sides are in states that they can connect to prevent packet loose
  if (sock_sync_data(res->sock, 1, "Q", &temp_char)) {
    // just send a dummy char back and forth
    fprintf(stderr, "sync error after QPs are were moved to RTS\n");
    rc = 1;
  }

connect_qp_exit:

  return rc;
}
```


-----------

+ `ibv_post_send`需要的信息

所有信息封装在`ibv_send_wr`结构体中；传送的数据相关信息封装在`ibv_sge`结构中；

```c
sge.addr = (uintptr_t)conn->send_region;
sge.length = BUFFER_SIZE;
sge.lkey = conn->send_mr->lkey;
```





