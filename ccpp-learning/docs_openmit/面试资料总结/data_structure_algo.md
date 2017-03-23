## 面试相关： 数据结构与算法

1. 求连续数组的最大差值？ 举例：

    ```c++
    input: [1,-3,-4,3,6,2,8,7]
    output: 12
    ```
    
2. 链表相关

    ```c++
    typedef struct ListNode {
        string data;
        ListNode * next;
    } LNode;
    ```
    
    + 链表反转

        `input: 1->2->3->4; output: 4->3->2->1`
        
    + 判断链表是否有环

        ```c++
        bool isExistLoop(LNode *pHead) {
            LNode * slow = pHead, *fast = pHead;
            while (fast && fast->next) {
                slow = slow->next;
                fast = fast->next->next;
                if (slow == fast) break;  
            }
            if (fast == NULL || fast->next == NULL) 
                return false;
            else
                return true;
        }
        ```
3. 树相关

    ```c++
    typedef struct Node {
        int data;
        Node *left;
        Node *right;
        Node(int x): data(x), left(NULL), right(NULL) {}
    } BTNode;
    ```
    
    + 寻找最近公共祖先节点

        ```c++
        bool father(BTNode *node1, BTNode *node2) {
            if (node1 == NULL) 
                return false;
            else if (node1 == node2) 
                return true;
            else 
                return father(node1->left, node2) || father(node1->right, node2);
        }
        
        void firstAncestor(BTNode * root, BTNode * n1, BTNode * n2, BTNode * out) {
            if (root == NULL || n1 == NULL || n2 == NULL) 
                return ;
            if (root && father(root, n1) && father(root, n2)) {
                out = root;
                firstAncestor(root->left, n1, n2, out);
                firstAncestor(root->right, n1, n2, out);
            }
        }
        ```
        
    + 二叉树的镜像

    
张爽面试结论：通过，可以招收培养

技术考察点：

+ 研究课题相关
    + 可以详细清楚地描述《音频编解码》研究课题，涉及的关键技术点理解比较透彻；
    + 对神经网络、深度学习等了解大概，并没有相关学习和实践经验；
+ 编程题
    + 求一棵二叉树的镜像；
    + 给定二叉树中任意两个节点，寻找其最近的公共祖先节点并返回；
    + 2道编程题都能完成的写出代码，思路清楚，调理清晰；
+ 评价
    + 优点： 
        + 思路清晰，理解问题反应比较快；
        + 编程能力不错，有Java/Python等工程经验；
        + 通过介绍之前的实习经历和技术，可以看出有不错的学习能力；
    + 缺点：
        + 数据挖掘、机器学习方向基本零基础，Hadoop/Spark相关没有经验，培养成本略高
    
－－－－－－－－－－－－－－－－
    
沈涛面试结论：通过

技术考察点：

+ 机器学习、项目相关考察
    + 对项目涉及到的算法，比如PCA等有比较好的认识，但搞不清楚降维算法的差异，比如PCA和SVD；
    + 熟悉svm／lr／决策树等常用模型，但认识比较浅，对常用的优化算法不清楚；
    + 公式推导：n维空间一个点到超平面（wx+b=0）的距离（考察svm中几何间隔），没能给出完整的推导过程；
    + 了解CNN的工作原理，但对其它DL模型认识比较浅；
+ 编程题
    + 求一棵二叉树的镜像；
    + 求单链表中倒数第k个节点（一次遍历）
    + 2道编程题都能完成的写出代码，思路清楚；
+ 评价
    + 有TensorFlow/Caffe使用和开发经验（在OpenMP和GPU suda上），研究可以与图像相关，与相关业务问题比较相关；
    + 熟悉C／C++／Python编程；
    + 对算法方向比较感兴趣，愿意从事数据挖掘、机器学习算法相关的工作；
    + 问题：对机器学习的模型和算法理解不够，问题不大（可以通过学习和训练来提升）；

－－－－－－－－－－

李香婷面试结论：不通过

技术考察点：

+ 机器学习、项目考察
    + 项目偏工程，对涉及到的算法和模型 不清楚；
    + 三个ML工作原理考察 均不理想：
        + svm优化目标表达式；
        + n维空间一个点到超平面（wx+b=0）的距离（考察svm中几何间隔），没能给出完整的推导过程；
        + lr参数求导公式推导；

+ 编程题：
    + 求单链表中倒数第k个节点（一次遍历），可以给出完整代码。
        
+ 评价
    + 机器学习方面基础比较弱：分不清svm/lr两个模型的差异；对其它常见的分类器只知其名，说不清楚其工作原理；
    + 偏向于暑期实习，对算法方向热情不高。


－－－－－－－－－－－－

高绍钧面试结论：不通过

技术考察点：

+ 机器学习、项目考察
    + 对项目涉及到的深度学习模型比如RNN、LSTM工作原理比较熟悉；  
    + 2个ML模型理解程度考察 均不理想：
        + n维空间一个点到超平面（wx+b=0）的距离（考察svm中几何间隔），没能给出完整的推导过程；
        + lr参数求导公式推导（以梯度法为例）；

+ 编程题：
    + 判断链表是否有环？
        
+ 评价
    + 机器学习方面基础比较弱，上过《机器学习》的课，但大部分都记不得了；
    + 语音之外的技术，比如文本、图像、数据处理、算法等均无经验，培养成本略高；
    + **因实验室有项目，最早5、6月份才能实习**。

