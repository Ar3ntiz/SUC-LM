import Foundation
import CoreGraphics
import CoreText

@main struct AttachmentTests {
    static func main() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func file(_ name: String, _ data: Data) throws -> URL { let url = root.appendingPathComponent(name); try data.write(to: url); return url }
        func rejects(_ action: () throws -> Void) { do { try action(); preconditionFailure("Invalid attachment accepted") } catch {} }
        func archive(_ name: String, entry: String, xml: String) throws -> URL {
            let base = root.appendingPathComponent(UUID().uuidString), path = base.appendingPathComponent(entry)
            try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(xml.utf8).write(to: path)
            let output = root.appendingPathComponent(name), p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/zip"); p.arguments = ["-q", "-r", output.path, "."]; p.currentDirectoryURL = base
            try p.run(); p.waitUntilExit(); precondition(p.terminationStatus == 0); return output
        }
        let unknown = try file("note.estensione-inventata", Data("Caffè e 日本語\n42".utf8))
        let read = try AttachmentReader.read(unknown); precondition(read.text.contains("日本語"))
        let utf16 = try file("unicode.bin", "Ciao UTF16".data(using: .utf16)!)
        let utf16Text = try AttachmentReader.read(utf16).text; precondition(utf16Text == "Ciao UTF16")
        let cp = try file("legacy.strana", "Caffè".data(using: .windowsCP1252)!)
        let cpText = try AttachmentReader.read(cp).text; precondition(cpText == "Caffè")
        let binary = try file("fake.txt", Data([0, 1, 2, 3, 255])); rejects { _ = try AttachmentReader.read(binary) }
        let empty = try file("empty", Data()); rejects { _ = try AttachmentReader.read(empty) }
        rejects { _ = try AttachmentReader.read(root) }
        let huge = try file("huge.txt", Data(repeating: 65, count: AttachmentReader.maxTextBytes + 1)); rejects { _ = try AttachmentReader.read(huge) }
        let tooBig = try file("oversize.txt", Data(repeating: 65, count: AttachmentReader.maxFileBytes + 1)); rejects { _ = try AttachmentReader.read(tooBig) }
        rejects { _ = try OfficeText.run("/usr/bin/yes", ["bounded output"], limit: 1024) }
        let word = """
        <?xml version="1.0"?><x:document xmlns:x="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><x:body><x:p><x:r><x:t>Ciao &amp; </x:t></x:r><x:r><x:t>mondo</x:t></x:r></x:p><x:tbl><x:tr><x:tc><x:p><x:r><x:t>Tabella</x:t></x:r></x:p></x:tc></x:tr></x:tbl></x:body></x:document>
        """
        let docx = try archive("word.qualsiasi", entry: "word/document.xml", xml: word)
        let docxText = try AttachmentReader.read(docx).text
        precondition(docxText.contains("Ciao & mondo") && docxText.contains("Tabella") && !docxText.contains("<x:"))
        let odf = """
        <?xml version="1.0"?><office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:z="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text><z:h>Titolo</z:h><z:p>Open<z:s z:c="2"/><z:span>Document</z:span><z:line-break/>fine</z:p></office:text></office:body></office:document-content>
        """
        let odt = try archive("open.odt", entry: "content.xml", xml: odf)
        let odtText = try AttachmentReader.read(odt).text
        precondition(odtText.contains("Titolo") && odtText.contains("Open  Document\nfine"))
        rejects { _ = try XMLText.extract(Data("<!DOCTYPE a [<!ENTITY x 'bad'>]><a>&x;</a>".utf8), kind: .word) }
        rejects { _ = try XMLText.extract(Data("<broken>".utf8), kind: .word) }
        let rtf = try file("rich.any", Data("{\\rtf1\\ansi Hello \\b world\\b0}".utf8))
        let rtfText = try AttachmentReader.read(rtf).text; precondition(rtfText.contains("Hello world"))
        let plain = try file("doc-source.txt", Data("Vecchio documento Word".utf8))
        let oldDoc = root.appendingPathComponent("legacy.doc")
        _ = try OfficeText.run("/usr/bin/textutil", ["-convert", "doc", "-output", oldDoc.path, plain.path], limit: 1024)
        let oldText = try AttachmentReader.read(oldDoc).text; precondition(oldText.contains("Vecchio documento Word"))
        // A real text PDF, including an empty page, exercises PDFKit extraction.
        let pdfData = NSMutableData(), consumer = CGDataConsumer(data: pdfData as CFMutableData)!
        var box = CGRect(x: 0, y: 0, width: 300, height: 300)
        let context = CGContext(consumer: consumer, mediaBox: &box, nil)!
        context.beginPDFPage(nil); context.textPosition = CGPoint(x: 20, y: 200)
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: "PDF readable text", attributes: [NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 12, nil)]))
        CTLineDraw(line, context); context.endPDFPage(); context.beginPDFPage(nil); context.endPDFPage(); context.closePDF()
        let pdf = try file("pdf.unknown", pdfData as Data)
        let pdfText = try AttachmentReader.read(pdf)
        precondition(pdfText.pages == 2 && pdfText.text.contains("PDF readable text") && pdfText.text.contains("senza testo"))
        var session = ChatSession(); session.draftAttachments = [read]
        session.messages = [ChatMessage(role: "user", text: "Riassumi", attachments: [read])]
        let roundtrip = try JSONDecoder().decode(ChatSession.self, from: JSONEncoder().encode(session))
        precondition(roundtrip.draftAttachments == [read] && roundtrip.messages[0].modelContent.contains("日本語"))
        precondition(!ChatMessage(role: "user", text: "", attachments: [read]).modelContent.isEmpty)
        var legacy = try JSONSerialization.jsonObject(with: JSONEncoder().encode(session)) as! [String: Any]
        legacy.removeValue(forKey: "draftAttachments")
        let migrated = try JSONDecoder().decode(ChatSession.self, from: JSONSerialization.data(withJSONObject: legacy))
        precondition(migrated.draftAttachments == nil)
        rejects { try AttachmentReader.validate(Array(repeating: read, count: 6)) }
        print("AttachmentTests: arbitrary extensions/encodings, binary/size rejection, DOCX/ODT XML, DOC/RTF, PDF, persistence and migration passed")
    }
}
