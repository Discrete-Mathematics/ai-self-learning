*来自https://github.com/unruledboy/DotNetStack/*
# DotNetStack
.NET技术栈，包括框架、SDK、工具等

![The Dot NET Stack](https://raw.githubusercontent.com/unruledboy/DotNetStack/master/preview.png)

# 什么是DotNetStack？以及为什么要了解它？
你是否曾经想过：
* .NET究竟包含哪些技术？
* 我掌握了多少？

我找不到一个真正全面的图表来展示.NET技术栈，所以我制作了自己版本的图表。

可能这里或那里有一些问题，比如分类、个别项目，但美在于你可以根据自己的需要修改它。

你可以在这里查看图形预览（使用鼠标移动/缩放）：

https://rawgit.com/unruledboy/DotNetStack/master/ux/DotNetStack.htm

# .NET技术栈

<!--BUILD_START-->

- [.NET](http://www.microsoft.com/net)
	- 公共语言基础设施（CLI）
	- 公共语言运行时（CLR）
		- 即时编译（JIT）
		- 垃圾回收器（GC）
			- 对象固定
			- 代
				- Gen 0
				- Gen 1
				- Gen 2
		- 内存管理
			- 栈
			- 托管堆
				- 大对象堆
				- 内存碎片
				- 堆整理
		- 安全性
	- 公共类型系统（CTS）
	- 公共语言规范（CLS）
	- 公共中间语言（CIL）
	- .NET框架
		- .NET框架
			- 1.0
			- 2.0
				- 泛型
			- 3.0
				- WPF
				- WCF
				- WF
			- 3.5
				- LINQ
			- 4.0
				- PLINQ
				- TPL
			- 4.5
				- 现代UI
				- 异步/等待（async/await）
			- 4.6
		- .NET Core
			- ASP.NET Core
		- Mono
		- DNX（.NET执行环境）
		- .NET Micro Framework（MF）
		- .NET Compact Framework（CF）
	- 基础类库（BCL）
		- 系统类型
		- 集合
		- 调试
		- 全球化
		- 输入输出（IO）
		- 安全性
		- 线程
		- 文本
	- 运行时基础库（RIL）
		- 系统
			- 应用程序域
			- 指针
			- 句柄
		- 反射
	- 框架类库（FCL）
		- ADO.NET
		- XML
		- 网络
		- 绘图
		- LINQ
			- 并行LINQ（PLINQ）
		- 互操作性（InterOp）
			- 平台调用（P/Invoke）
		- 任务并行库（TPL）
		- 序列化
	- 核心概念
		- 应用程序域
		- 程序集
		- 命名空间
		- 代码访问安全（CAS）
		- 泛型
		- 全局程序集缓存（GAC）
		- 强命名
		- 弱引用
		- 事件模型
		- 基于任务的异步模型
			- 异步/等待（async/await）
	- 核心特性
		- 语言独立性
		- 互操作性
		- 可移植性
	- 核心对象
		- 类
		- 函数
		- 属性
		- 字段
		- 事件
		- 类型
			- 引用类型
			- 值类型
	- CLI语言
		- [C#](https://msdn.microsoft.com/en-us/library/67ef8sbd.aspx)
		- [VB.NET](https://msdn.microsoft.com/en-us/library/2x7h1hfk.aspx)
		- [F#](http://fsharp.org/)
		- [C++/CLI](https://en.wikipedia.org/wiki/C%2B%2B/CLI)
		- [ClojureCLR](https://github.com/clojure/clojure-clr)
		- [Fantom](http://fantom.org/)
		- [Visual COBOL](http://www.microfocus.com/products/micro-focus-developer/visual_cobol/)
	- 动态语言运行时（DLR）
		- DLR语言
			- IronPython
			- IronRuby
			- IronScheme
			- IronLisp
	- 桌面
		- 图形界面（GUI）
			- Windows Forms
			- Windows Presentation Foundation（WPF）
				- XAML
		- 命令行界面（CUI）
			- 控制台
		- Windows服务
	- Web
		- ASP.NET
			- ASP.NET运行时
				- HttpHandler
				- HttpModule
			- ASP.NET Web页面
			- ASP.NET Web表单
				- 核心对象/概念
					- Web表单
					- Web控件
					- 回发
					- 会话
					- 视图状态
				- Ajax控制工具包
			- ASP.NET MVC
				- MVC
					- 模型
					- 视图
					- 控制器
				- 视图引擎
					- Razor
					- Web表单
					- 自定义
				- 核心对象/概念
					- 路由
					- 控制器
					- 动作
					- 过滤器
		- Silverlight
		- SharePoint
		- IIS
	- SOA
		- ASP.NET Web API
			- REST（表现状态转移）
			- CRUD
				- CREATE: POST
				- RETRIEVE: GET
				- UPDATE: PUT
				- DELETE: DELETE
			- 消费者
				- HttpClient
				- RestSharp
			- 媒体格式化器
				- XML
				- JSON
				- BSON
				- 自定义
			- 核心概念
				- 按约定
				- 路由
				- 内容协商
				- 模型验证
				- 模型绑定
				- 托管
			- 核心对象
				- API控制器
				- 动作
				- HTTP消息处理器
				- 过滤器
		- ASP.NET Web服务
			- 核心对象
				- WebService
				- WebMethod
			- 文件类型
				- XML（可扩展标记语言）
				- SOAP（简单对象访问协议）
				- WSDL（Web服务描述语言）
				- UDDI（通用描述、发现和集成）
		- Windows通信基础（WCF）
			- ABC
				- 地址
				- 绑定
				- 合同
			- 绑定
				- BasicHttpBinding
				- WsHttpBinding
				- WsDualHttpBinding
				- WsFederationHttpBinding
				- NetNamedPipeBinding
				- NetTcpBinding
				- NetPeerTcpBinding
				- NetMsmqBinding
			- 合同
				- 数据合同
				- 服务合同
				- 操作合同
				- 消息合同
		- [AppFabric](https://en.wikipedia.org/wiki/AppFabric)
			- 持久化
			- 托管
			- 监控
			- 缓存
		- .NET远程调用
	- [Open Web接口（OWIN）](http://owin.org/)
		- SignalR
		- Nancy
		- Katana
	- SDK
		- [Windows工作流基础（WWF）](https://msdn.microsoft.com/en-us/vstudio/jj684582.aspx)
		- [托管可扩展性框架（MEF）](https://msdn.microsoft.com/en-us/library/dd460648(v=vs.110).aspx)
		- [Reactive Extensions（Rx）](https://github.com/Reactive-Extensions)
		- 企业库 ;-)
	- 开发
		- [Visual Studio](https://www.visualstudio.com/)
		- [Visual Studio Online](https://www.visualstudio.com/en-us/products/what-is-visual-studio-online-vs.aspx)
		- [Visual Studio Code](https://code.visualstudio.com/)
		- [Blend](https://msdn.microsoft.com/en-us/library/jj171012.aspx)
		- [LightSwitch](https://msdn.microsoft.com/en-us/library/lightswitch.aspx)
		- [MonoDevelop](http://www.monodevelop.com/)
		- [SharpDevelop](http://www.icsharpcode.net/OpenSource/SD/Default.aspx)
	- 平台
		- [Windows Runtime（WinRT）](https://en.wikipedia.org/wiki/Windows_Runtime)
		- [Xamarin](http://xamarin.com/)
		- [Universal Apps](https://dev.windows.com/en-us/develop/build-apps-shared-code)
			- WinJS
		- [.NET Native](https://msdn.microsoft.com/en-us/vstudio/dotnetnative.aspx)
		- Windows Phone SDK
		- [Azure](https://azure.microsoft.com)
	- 生产力
		- [Team Foundation Server（TFS）](https://www.visualstudio.com/en-us/products/tfs-overview-vs.aspx)
			- 产品待办事项
			- 任务
		- [NuGet](https://www.nuget.org/)
		- [Chocolatey](https://chocolatey.org/)
		- [PowerShell](https://en.wikipedia.org/wiki/Windows_PowerShell)
	- 构建
		- MSBuild
		- CodeDom
		- [Roslyn](https://github.com/dotnet/roslyn)
		- [CruiseControl.NET](http://cruisecontrol.net/)
		- csc/vbc/fsc
	- 测试
		- MSTest
			- TestClass
			- TestMethod
		- [NUnit](http://www.nunit.org/)
			- TestFixture
			- Test
		- Mock
			- [Moq](https://github.com/Moq/)
		- Fakes
		- Microsoft Test Manager
	- 部署/发布
		- ClickOnce
		- Web/FTP
		- Cloud
		- 文件
	- 框架（Fx）工具
		- Ildasm（MSIL反汇编器）
		- NGen（本地生成，提前编译）
		- sn（强命名）
		- gacutil
		- [FxCop](https://en.wikipedia.org/wiki/FxCop)
	- [数据](https://github.com/unruledboy/DatabaseStack)
		- ADO.NET
		- 对象/关系映射（O/RM）
			- [Entity Framework（EF）](https://github.com/aspnet/EntityFramework)
			- [NHibernate](http://nhibernate.info/)
			- [AutoMapper](http://automapper.org/)
		- NoSQL
			- [RavenDB](http://ravendb.net/)
			- [Azure Document DB](http://azure.microsoft.com/en-us/services/documentdb/)
		- 缓存
			- [NCache](http://www.alachisoft.com/ncache/)
	- 组件
		- 依赖注入（IoC）
			- [Autofac](http://autofac.org/)
			- [Unity](https://github.com/unitycontainer/unity)
			- [NInject](http://www.ninject.org/)
			- [Windsor](https://github.com/castleproject/Windsor/blob/master/docs/README.md)
		- 消息/队列
			- [NServiceBus](http://particular.net/nservicebus)
			- [RabbitMQ](https://www.rabbitmq.com/)
			- [MassTransit](http://masstransit-project.com/)
			- [NetMQ](https://github.com/zeromq/netmq)
		- 调度
			- [Quartz.NET](http://www.quartz-scheduler.net/)

<!--BUILD_END-->

