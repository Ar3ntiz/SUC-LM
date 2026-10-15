import Foundation
@main struct MarkdownTests {
    static func main() {
        let blocks=ChatMarkdown.parse("# Title\n\n1. First\n  - [x] Done\n> Quote\n\n| A | B |\n| --- | ---: |\n| one | two |\n\n~~~swift\nlet x = 1\n~~~\n\nText `inline``` code`\n---")
        assert(blocks[0].kind == .heading(1));assert(blocks[1].kind == .list("1.",0));assert(blocks[2].kind == .list("☑",1));assert(blocks[3].kind == .quote)
        assert(blocks[4].kind == .table(["A","B"],["left","right"]));assert(blocks[4].rows == [["one","two"]]);assert(blocks[5].kind == .code("swift"));assert(blocks[5].text == "let x = 1");assert(blocks[6].kind == .paragraph);assert(blocks[7].kind == .rule)
        assert(ChatMarkdown.parse("```python\npartial").first?.text == "partial")
        assert(ChatMarkdown.cells("| a\\|b | `x|y` |") == ["a\\|b","`x|y`"])
        let inline=ChatMarkdown.inline("**bold** [web](https://example.com) [bad](file:///tmp/a) https://example.org")
        let links=inline.runs.compactMap{$0.link?.absoluteString};assert(links.contains("https://example.com"));assert(links.contains("https://example.org"));assert(!links.contains(where:{$0.hasPrefix("file:")}))
        assert(inline.runs.contains{$0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true})
        assert(!ChatMarkdown.webURL(URL(string:"javascript:alert(1)")!))
        print("Markdown tests passed: blocks, tables, lists, streaming fences, inline styles and safe links")
    }
}
