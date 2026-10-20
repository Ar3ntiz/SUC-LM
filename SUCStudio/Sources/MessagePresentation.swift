import Foundation

struct ThinkingParts:Equatable {
    var reasoning:String
    var answer:String
    var active:Bool
    static func split(_ text:String,separate:String?=nil,implicit:Bool=false)->Self {
        if let separate,!separate.isEmpty {return Self(reasoning:separate,answer:text,active:text.isEmpty)}
        let trimmed=text.trimmingCharacters(in:.whitespacesAndNewlines)
        let tag=trimmed.hasPrefix("<analysis>") || "<analysis>".hasPrefix(trimmed) ? "analysis":"think"
        let open="<"+tag+">",close="</"+tag+">"
        if !trimmed.isEmpty,open.hasPrefix(trimmed),trimmed != open {return Self(reasoning:"",answer:"",active:true)}
        let explicit=trimmed.hasPrefix(open)
        if explicit || implicit {
            let body=explicit ? String(trimmed.dropFirst(open.count)):trimmed
            if let end=body.range(of:close){return Self(reasoning:String(body[..<end.lowerBound]),answer:String(body[end.upperBound...]).trimmingCharacters(in:.whitespacesAndNewlines),active:false)}
            var reasoning=body
            for size in stride(from:min(close.count-1,body.count),through:1,by:-1) {if body.hasSuffix(String(close.prefix(size))){reasoning=String(body.dropLast(size));break}}
            return Self(reasoning:reasoning,answer:"",active:true)
        }
        return Self(reasoning:"",answer:text,active:false)
    }
}
enum MessagePresentation {
    static let previewLimit=8000
    static func preview(_ text:String)->String {String(text.prefix(previewLimit))}
    static func isLong(_ text:String)->Bool{text.utf8.count>previewLimit && text.count>previewLimit}
    static func isHTML(_ text:String)->Bool{
        let head=text.prefix(1024).lowercased()
        return head.contains("<!doctype html") || head.contains("<html") || head.contains("<head>") || head.contains("<body")
    }
    static func codeFence(_ source:String,language:String)->String {
        var longest=0,run=0
        for c in source{if c=="`"{run+=1;longest=max(longest,run)}else{run=0}}
        let marker=String(repeating:"`",count:max(3,longest+1))
        let label=String(language.filter{$0.isLetter || $0.isNumber || $0=="-" || $0=="_"}.prefix(30))
        return marker+label+"\n"+source+"\n"+marker
    }
}
struct PreparedMarkdown {
    var blocks:[MarkdownBlock]
    var formatted:[String:AttributedString]
    static func make(_ text:String)->Self {
        let source=MessagePresentation.preview(text)
        let blocks=MessagePresentation.isHTML(source) ? [MarkdownBlock(kind:.code("html"),text:source)]:Array(ChatMarkdown.parse(source).prefix(200))
        var formatted:[String:AttributedString]=[:]
        for block in blocks {
            switch block.kind {
            case .code,.rule:break
            case .table(let headers,_):for cell in headers+block.rows.flatMap({$0}) {if formatted[cell]==nil{formatted[cell]=ChatMarkdown.inline(cell)}}
            default:if formatted[block.text]==nil{formatted[block.text]=ChatMarkdown.inline(block.text)}
            }
        }
        return Self(blocks:blocks,formatted:formatted)
    }
}


/// A fixed start index keeps older visible rows stable while new tokens/messages arrive.
enum TranscriptWindow {
    static func initial(_ count:Int)->Int {max(0,count-80)}
    static func updated(start:Int,count:Int,following:Bool)->Int {
        following && count-start>160 ? initial(count):min(start,count)
    }
    static func earlier(_ start:Int)->Int {max(0,start-40)}
}
