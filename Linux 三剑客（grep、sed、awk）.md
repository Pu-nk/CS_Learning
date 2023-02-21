# 正则表达式
## 基本正则表达式（BRE）
| 符号   | 作用                                   |
| ------ | -------------------------------------- |
| ^      | 匹配模式的开头，以...开头              |
| $      | 匹配模式的结尾，以...结尾              |
| ^$     | 匹配空行                               |
| .      | 匹配任意一个且只有一个字符             |
| \\     | 转义字符                               |
| \*     | 匹配一个字符0次或者多次                |
| .\*    | 匹配所有的内容                         |
| ^.\*   | 匹配以任意多个字符开头的内容           |
| .\$    | 匹配以任意多个字符开头的内容           |
| [abc]  | 匹配中括号内任意一个字符a/b/c          |
| [\^abc] | 匹配中除了括号内任意一个字符以外的内容 |
## 扩展正则表达式（ERE）
```ad-note
扩展正则必须用`grep -E`才能生效
```
| 符号   | 作用                       |
| ------ | -------------------------- |
| +      | 匹配前一个字符一次或者多次 |
| [:/]+  | 匹配:或者/一次或者多次     |
| ？     | 匹配前一个字符零或一次     |
| ｜     | 或者,a\|b                  |
| ()     | 分组过滤                   |
| a{n,m} | 匹配a最少n次,最多m次       |
| a{n,}  | 匹配a最少n次               |
| a{n,m} | 匹配a最多m次               |
| a{}n   | 匹配a正好n次               |
# grep
定义：global search REgular expression and Print out the line
```
grep [options] [pattern] file
		-i: 忽略大小写
		-o: 仅显示能够匹配到的字符串
		-v: 显示不能被匹配到的行
		-n: 显示匹配行号
		-c: 统计行数
```
- 一个简单的案例
```bash
grep -i 'root' pwd.txt -n
grep -i '^$' pwd.txt - v | greq -i '^#' pwd.txt -v
# 贪婪匹配
grep -n '.*e' pwd.txt 
# 全文匹配
grep '[a-z0-5]' pwd.txt -n
grep '[^0-5]' pwd.txt -n

# 扩展正则
grep -E 'i+' pwd.txt 
grep -E 'go?d' pwd.txt

# 在目录下找到txt
find /data -name "*.txt" | grep -E 'a|l'

# 分组查询
grep -E  "g(oo|la)d" test.txt
grep -E "(l..e).*\1" test.txt  # 表示后面引用前面的匹配条件
```
