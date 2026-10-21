import Foundation

/// The app preference is independent of the operating-system language.
/// Only application-owned strings pass through this catalog; chat content and wire names do not.
struct StudioTextKey: ExpressibleByStringLiteral, ExpressibleByStringInterpolation {
    let key: String
    let arguments: [String]
    init(stringLiteral value: String) { key=value; arguments=[] }
    init(stringInterpolation value: StringInterpolation) { key=value.key; arguments=value.arguments }
    struct StringInterpolation: StringInterpolationProtocol {
        var key="", arguments=[String]()
        init(literalCapacity: Int, interpolationCount: Int) { key.reserveCapacity(literalCapacity); arguments.reserveCapacity(interpolationCount) }
        mutating func appendLiteral(_ literal: String) { key+=literal }
        mutating func appendInterpolation<T>(_ value: T) { key+="{\(arguments.count)}"; arguments.append(String(describing:value)) }
    }
}
func L(_ value: StudioTextKey) -> String { StudioL10n.render(value.key,arguments:value.arguments) }
func LT(_ value: String) -> String { StudioL10n.translate(value) }
enum StudioL10n {
    private static let lock=NSLock()
    private static var selected="en"
    static var language:String {
        get { lock.lock();defer{lock.unlock()};return selected }
        set { lock.lock();selected=normalized(newValue);lock.unlock() }
    }
    static func normalized(_ value:String?) -> String { value == "it" ? "it":"en" }
    private static let placeholderRegex=try! NSRegularExpression(pattern:#"\{(\d+)\}"#)
    static func render(_ key:String,arguments:[String]=[]) -> String {
        let text=language == "it" ? (StudioCatalog.italian[key] ?? key):(StudioCatalog.english[key] ?? key)
        // Replace placeholders in one pass: substituted file names/text cannot become placeholders.
        let regex=placeholderRegex
        let source=text as NSString
        var result="", cursor=0
        for match in regex.matches(in:text,range:NSRange(location:0,length:source.length)) {
            result+=source.substring(with:NSRange(location:cursor,length:match.range.location-cursor))
            let index=Int(source.substring(with:match.range(at:1))) ?? -1
            result+=arguments.indices.contains(index) ? arguments[index]:source.substring(with:match.range)
            cursor=NSMaxRange(match.range)
        }
        return result+source.substring(from:cursor)
    }
    private static let templates:[(String,NSRegularExpression)] = StudioCatalog.english.keys.filter{$0.contains("{0}")}.sorted{$0.count > $1.count}.compactMap { key in
        let parts=key.components(separatedBy:try! NSRegularExpression(pattern:#"\{\d+\}"#))
        let pattern="\\A"+parts.map{NSRegularExpression.escapedPattern(for:$0)}.joined(separator:"([\\s\\S]*?)")+"\\z"
        return (try? NSRegularExpression(pattern:pattern)).map{(key,$0)}
    }
    static func report(_ text:String) -> String {
        text.components(separatedBy:"\n").map { line in
            if StudioCatalog.english[line] != nil { return translate(line) }
            return line.components(separatedBy:" · ").map{translate($0)}.joined(separator:" · ")
        }.joined(separator:"\n")
    }
    static func translate(_ text:String) -> String {
        if StudioCatalog.english[text] != nil {return render(text)}
        for (key,regex) in templates {
            let source=text as NSString
            if let match=regex.firstMatch(in:text,range:NSRange(location:0,length:source.length)) {
                return render(key,arguments:(1..<match.numberOfRanges).map{source.substring(with:match.range(at:$0))})
            }
        }
        return text
    }
}
private extension String {
    func components(separatedBy regex:NSRegularExpression) -> [String] {
        let source=self as NSString; var result=[String](),cursor=0
        for match in regex.matches(in:self,range:NSRange(location:0,length:source.length)) {
            result.append(source.substring(with:NSRange(location:cursor,length:match.range.location-cursor)));cursor=NSMaxRange(match.range)
        }
        result.append(source.substring(from:cursor));return result
    }
}
