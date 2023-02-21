Author: Zehoo Pu
Created: January 27, 2022 2:27 PM
Last edit time: February 6, 2022 10:53 AM
Tags: git, programming, 杂记
link: https://zhuanlan.zhihu.com/p/135183491

- 创建一个项目，在命令行输入`git init` 初始化项目，这时候当前目录下会多了一个.git的目录，这个目录是Git用来跟踪管理版本。
- 添加文件到暂存区：
    - 如果添加当前目录下的所有文件，就用 `git add .`
    - 如果添加指定的文件，就用 `git add filename`
- 提交文件输入 `git commit -m '备注'`
- 使用 `git status` 命令来查看文件状态
- 查看历史日志 `git log / git log --pretty==online`
- 版本回退至上一个版本使用 `git reset --hard HEAD^`
- 版本回退至上上个版本使用 `git reset --hard HEAD^^`
- 版本回退至前n个版本 `git reset --hard HEAD~n`
- 回退至指定版本 `git reset --hard 版本号` ，版本号可以通过`git reflog` 来获取
- `git checkout --file` 丢弃工作区的更改
- `git remote add origin[https://github.com/tugenhua0707/testgit.git](https://github.com/tugenhua0707/testgit.git)` 添加github仓库
- `git push` 推送master分支，第一次推送master分支时，加上了 –u参数，Git不但会把本地的master分支内容推送的远程新的master分支，还会把本地的master分支和远程的master分支关联起来，在以后的推送或者拉取时就可以简化命令。推送成功后，可以立刻在github页面中看到远程库的内容已经和本地一模一样了，从现在起，只要本地作了提交，就可以通过如下命令：`git push -u origin master` ，origin其实是一个用来进行配置的hostname
- `git clone`克隆一个本地库
- 创建dev分支`git branch name`，切换并创建dev分支 `git checkout -b dev`
- 切换到主分支，合并dev分支 `git merge dev`
- 删除分支 `git branch -d name`
- 通常合并分支时，git一般使用`”Fast forward”`模式，在这种模式下，删除分支后，会丢掉分支信息，现在我们来使用带参数 `–no-ff`来禁用”Fast forward”模式
- 要查看远程库的详细信息使用 `git remote –v`
- 抓取分支 `git pull`

- 要查看远程库的信息使用 `git remote`