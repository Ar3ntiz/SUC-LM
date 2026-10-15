import Foundation

struct MarkdownBlock: Equatable {
    enum Kind: Equatable {case paragraph, heading(Int), list(String,Int), quote, code(String), table([String],[String]), rule}
    var kind:Kind
    var text:String
    var rows:[[String]] = []
}
enum ChatMarkdown {
    static func cells(_ line:String)->[String] {
        var text=line.trimmingCharacters(in:.whitespaces);if text.first=="|"{text.removeFirst()};if text.last=="|"{text.removeLast()}
        var result:[String]=[],cell="",escaped=false,inCode=false
        for ch in text {
            if escaped{cell.append(ch);escaped=false;continue}
            if ch=="\\"{cell.append(ch);escaped=true;continue}
            if ch=="`"{inCode.toggle()}
            if ch=="|" && !inCode{result.append(cell.trimmingCharacters(in:.whitespaces));cell=""}else{cell.append(ch)}
        }
        result.append(cell.trimmingCharacters(in:.whitespaces));return result
    }
    static func parse(_ source:String)->[MarkdownBlock] {
        let lines=source.replacingOccurrences(of:"\r\n",with:"\n").components(separatedBy:"\n")
        var blocks:[MarkdownBlock]=[],paragraph:[String]=[],i=0
        func flush(){if !paragraph.isEmpty{blocks.append(MarkdownBlock(kind:.paragraph,text:paragraph.joined(separator:"\n")));paragraph=[]}}
        func match(_ pattern:String,_ line:String)->[String]? {
            guard let regex=try? NSRegularExpression(pattern:pattern),let m=regex.firstMatch(in:line,range:NSRange(line.startIndex...,in:line)) else{return nil}
            return (0..<m.numberOfRanges).map{Range(m.range(at:$0),in:line).map{String(line[$0])} ?? ""}
        }
        while i<lines.count {
            let line=lines[i],trim=line.trimmingCharacters(in:.whitespaces)
            if let m=match("^ {0,3}(`{3,}|~{3,})(.*)$",line) {
                flush();let marker=m[1],language=m[2].trimmingCharacters(in:.whitespaces);i+=1;var code:[String]=[]
                while i<lines.count {
                    let close=lines[i].trimmingCharacters(in:.whitespaces)
                    if close.count>=marker.count && close.allSatisfy({$0==marker.first!}){i+=1;break}
                    code.append(lines[i]);i+=1
                }
                blocks.append(MarkdownBlock(kind:.code(language),text:code.joined(separator:"\n")));continue
            }
            if trim.isEmpty{flush();i+=1;continue}
            if i+1<lines.count,line.contains("|") {
                let separators=cells(lines[i+1])
                if !separators.isEmpty,separators.allSatisfy({$0.range(of:"^:?-{3,}:?$",options:.regularExpression) != nil}) {
                    flush();let headers=cells(line),align=separators.map{$0.hasPrefix(":") && $0.hasSuffix(":") ? "center":$0.hasSuffix(":") ? "right":"left"};i+=2;var rows:[[String]]=[]
                    while i<lines.count,lines[i].contains("|"),!lines[i].trimmingCharacters(in:.whitespaces).isEmpty{rows.append(cells(lines[i]));i+=1}
                    blocks.append(MarkdownBlock(kind:.table(headers,align),text:"",rows:rows));continue
                }
            }
            if let m=match("^ {0,3}(#{1,6})[ \\t]+(.*)$",line){flush();blocks.append(MarkdownBlock(kind:.heading(m[1].count),text:m[2].replacingOccurrences(of:"[ \\t]+#+[ \\t]*$",with:"",options:.regularExpression)));i+=1;continue}
            if trim.range(of:"^(?:\\*\\s*){3,}$|^(?:-\\s*){3,}$|^(?:_\\s*){3,}$",options:.regularExpression) != nil{flush();blocks.append(MarkdownBlock(kind:.rule,text:""));i+=1;continue}
            if let m=match("^( *)([-+*]|[0-9]+[.)])[ \\t]+(.*)$",line){
                flush();var text=m[3],marker=m[2].count==1 ? "•":m[2]
                if text.hasPrefix("[ ] "){marker="☐";text=String(text.dropFirst(4))}
                else if text.lowercased().hasPrefix("[x] "){marker="☑";text=String(text.dropFirst(4))}
                blocks.append(MarkdownBlock(kind:.list(marker,min(8,m[1].count/2)),text:text));i+=1;continue
            }
            if trim.hasPrefix(">") {flush();var quotes:[String]=[]
                while i<lines.count,lines[i].trimmingCharacters(in:.whitespaces).hasPrefix(">") {var value=String(lines[i].trimmingCharacters(in:.whitespaces).dropFirst());if value.first==" "{value.removeFirst()};quotes.append(value);i+=1}
                blocks.append(MarkdownBlock(kind:.quote,text:quotes.joined(separator:"\n")));continue
            }
            paragraph.append(line);i+=1
        }
        flush();return blocks
    }
    static func webURL(_ url:URL)->Bool{["http","https"].contains(url.scheme?.lowercased() ?? "") && url.host != nil}
    static func inline(_ text:String)->AttributedString {
        var result=(try? AttributedString(markdown:text,options:.init(interpretedSyntax:.inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
        for run in result.runs {if let url=run.link,!webURL(url){result[run.range].link=nil}}
        let plain=String(result.characters)
        if let detector=try? NSDataDetector(types:NSTextCheckingResult.CheckingType.link.rawValue) {
            for match in detector.matches(in:plain,range:NSRange(plain.startIndex...,in:plain)) {
                guard let url=match.url,webURL(url),let range=Range(match.range,in:plain),let lower=AttributedString.Index(range.lowerBound,within:result),let upper=AttributedString.Index(range.upperBound,within:result),!result[lower..<upper].runs.contains(where:{$0.link != nil}) else{continue}
                result[lower..<upper].link=url
            }
        }
        return result
    }
}
