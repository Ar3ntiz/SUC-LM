import Foundation
import PDFKit
import Darwin

// Local harness: content sniffing, PDFKit and bounded macOS converters. No network or model calls.
enum AttachmentReader {
    static let maxFiles = 5
    static let maxFileBytes = 10 * 1024 * 1024
    static let maxTextBytes = 256 * 1024
    static let maxTotalTextBytes = 512 * 1024
    static func failure(_ message: String) -> NSError {
        NSError(domain: "SUCStudio.Attachments", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    static func validate(_ files: [ChatAttachment]) throws {
        guard files.count <= maxFiles, files.reduce(0, { $0 + $1.textBytes }) <= maxTotalTextBytes,
              files.reduce(0, { $0 + ($1.imageData?.count ?? 0) }) <= maxTotalImageBytes,
              files.allSatisfy({ $0.isImage ? validateImage($0) : (!$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.textBytes <= maxTextBytes) }) else {
            throw failure("Massimo 5 allegati, 256 KiB di testo per file e 512 KiB complessivi; immagini JPEG convertite fino a 2 MiB ciascuna, 8 MiB totali. Dividi i documenti in parti più piccole.")
        }
    }
    static func decodeText(_ data: Data) throws -> String {
        let encoding: String.Encoding
        if data.starts(with: [0xFF,0xFE,0,0]) || data.starts(with: [0,0,0xFE,0xFF]) { encoding = .utf32 }
        else if data.starts(with: [0xFF,0xFE]) || data.starts(with: [0xFE,0xFF]) { encoding = .utf16 }
        else { encoding = .utf8 }
        let decoded = String(data: data, encoding: encoding) ?? (encoding == .utf8 ? String(data: data, encoding: .windowsCP1252) : nil)
        guard var text = decoded else { throw failure("Non è stato trovato testo leggibile. Usa UTF-8, UTF-16/32 con BOM o Windows-1252.") }
        if text.first == "\u{FEFF}" { text.removeFirst() }
        guard !text.unicodeScalars.contains(where: { ($0.value < 32 && ![9, 10, 12, 13].contains($0.value)) || (127...159).contains($0.value) }) else {
            throw failure("File binario senza un estrattore compatibile. Sono ammessi testi con qualsiasi estensione, Word, OpenDocument e PDF testuali; le scansioni PDF richiedono OCR.")
        }
        return text
    }
    static func read(_ url: URL) throws -> ChatAttachment {
        guard url.isFileURL else { throw failure("Sono accettati soltanto file locali.") }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true else { throw failure("Trascina file singoli, non cartelle o pacchetti.") }
        guard let size = values.fileSize, size <= maxFileBytes else { throw failure("Il file supera il limite di 10 MiB.") }
        // Bounded read also covers files that grow after the initial size check.
        let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
        let data = try handle.read(upToCount: maxFileBytes + 1) ?? Data()
        guard data.count <= maxFileBytes else { throw failure("Il file supera il limite di 10 MiB.") }
        let imageName=String(url.lastPathComponent.components(separatedBy:.controlCharacters).joined(separator:" ").prefix(200))
        if let image=try readImage(data,name:imageName) { try validate([image]);return image }
        var text = "", pages: Int?
        if data.starts(with: Data("%PDF-".utf8)) {
            guard let document = PDFDocument(data: data), !document.isLocked else { throw failure("PDF non leggibile o protetto da password.") }
            guard document.pageCount <= 200 else { throw failure("PDF troppo lungo (massimo 200 pagine).") }
            pages = document.pageCount
            var parts: [String] = [], bytes = 0, emptyPages = 0
            for index in 0..<document.pageCount {
                let pageText = document.page(at: index)?.string ?? ""
                if pageText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { emptyPages += 1; continue }
                let part = "[Pagina \(index + 1)]\n" + pageText
                bytes += part.utf8.count + 2
                guard bytes <= maxTextBytes else { throw failure("Il testo del PDF supera 256 KiB. Dividi il documento; nessun testo viene troncato.") }
                parts.append(part)
            }
            guard !parts.isEmpty else { throw failure("Il PDF non contiene testo estraibile. Le scansioni richiedono OCR, non disponibile nell’app.") }
            // Tell the model/user which pages have no text instead of silently claiming full extraction.
            if emptyPages > 0 { parts.insert("[\(emptyPages) pagine senza testo estraibile: immagini/scansioni o pagine vuote non vengono lette.]", at: 0) }
            text = parts.joined(separator: "\n\n")
        } else if data.starts(with: [0x50, 0x4B, 0x03, 0x04]) {
            text = try OfficeText.extractArchive(data)
        } else if data.starts(with: [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
            text = try OfficeText.convert(data, format: "doc")
        } else if data.starts(with: Data("{\\rtf".utf8)) {
            text = try OfficeText.convert(data, format: "rtf")
        } else {
            text = try decodeText(data)
        }
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = String(url.lastPathComponent.components(separatedBy: .controlCharacters).joined(separator: " ").prefix(200))
        let result = ChatAttachment(name: name, text: text, originalBytes: data.count, pages: pages)
        try validate([result]); return result
    }
}

// Never unpack an archive to disk. Read only selected XML parts through unzip -p,
// cap decompression output and time, and reject DTD/entity declarations.
enum OfficeText {
    static func run(_ executable: String, _ arguments: [String], limit: Int) throws -> Data {
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        process.standardOutput = pipe; process.standardError = FileHandle.nullDevice; process.standardInput = FileHandle.nullDevice
        try process.run()
        let timeout = DispatchWorkItem {
            if process.isRunning { process.terminate() }
            DispatchQueue.global().asyncAfter(deadline: .now() + 1) { if process.isRunning { kill(process.processIdentifier, SIGKILL) } }
        }
        DispatchQueue.global().asyncAfter(deadline: .now() + 10, execute: timeout)
        defer {
            timeout.cancel()
            if process.isRunning { kill(process.processIdentifier, SIGKILL); process.waitUntilExit() }
            try? pipe.fileHandleForReading.close()
        }
        var result = Data()
        while let chunk = try pipe.fileHandleForReading.read(upToCount: 8192), !chunk.isEmpty {
            if result.count + chunk.count > limit {
                if process.isRunning { kill(process.processIdentifier, SIGKILL) }
                process.waitUntilExit()
                throw AttachmentReader.failure("Documento troppo complesso o troppo grande dopo l’estrazione. Dividilo in parti più piccole.")
            }
            result.append(chunk)
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw AttachmentReader.failure("Documento non leggibile, protetto, non supportato o conversione oltre 10 secondi.") }
        return result
    }
    static func withFile<T>(_ data: Data, _ body: (URL) throws -> T) throws -> T {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("SUC-attachment-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("document")
        try data.write(to: file); try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        return try body(file)
    }
    static func convert(_ data: Data, format: String) throws -> String {
        try withFile(data) { file in
            let output = try run("/usr/bin/textutil", ["-convert", "txt", "-format", format, "-encoding", "UTF-8", "-stdout", file.path], limit: AttachmentReader.maxTextBytes)
            return try AttachmentReader.decodeText(output)
        }
    }
    static func extractArchive(_ data: Data) throws -> String {
        try withFile(data) { file in
            let listing = try run("/usr/bin/unzip", ["-Z", "-1", file.path], limit: 128 * 1024)
            guard let names = String(data: listing, encoding: .utf8) else { throw AttachmentReader.failure("Indice del documento non leggibile.") }
            let entries = names.split(separator: "\n").map(String.init)
            var selected: [String] = [], kind: XMLText.Kind
            if entries.contains("word/document.xml") {
                kind = .word; selected = ["word/document.xml"]
                selected += entries.filter { $0.range(of: #"^word/(header[0-9]*|footer[0-9]*|footnotes|endnotes)\.xml$"#, options: .regularExpression) != nil }.sorted()
            } else if entries.contains("content.xml") {
                kind = .openDocument; selected = ["content.xml"]
            } else { throw AttachmentReader.failure("Archivio non riconosciuto come Word/OpenDocument. Estrai i singoli file testuali e allegali.") }
            guard selected.count <= 32 else { throw AttachmentReader.failure("Documento con troppe parti XML (massimo 32).") }
            var parts: [String] = [], total = 0
            for entry in selected {
                let xml = try run("/usr/bin/unzip", ["-p", file.path, entry], limit: 4 * 1024 * 1024)
                let value = try XMLText.extract(xml, kind: kind)
                if !value.isEmpty { total += value.utf8.count + 2; guard total <= AttachmentReader.maxTextBytes else { throw AttachmentReader.failure("Testo estratto oltre 256 KiB; nessun taglio automatico.") }; parts.append(value) }
            }
            return parts.joined(separator: "\n\n")
        }
    }
}

final class XMLText: NSObject, XMLParserDelegate {
    enum Kind { case word, openDocument }
    let kind: Kind
    var result = "", capture = 0, bytes = 0, oversized = false
    init(kind: Kind) { self.kind = kind }
    static func extract(_ data: Data, kind: Kind) throws -> String {
        // XMLParser's external resolution is disabled, but also reject internal DTDs.
        let source = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .utf16) ?? ""
        guard !source.isEmpty, !source.contains("<!DOCTYPE"), !source.contains("<!ENTITY") else { throw AttachmentReader.failure("XML non supportato: codifica o dichiarazioni DTD/entità.") }
        let reader = XMLText(kind: kind), parser = XMLParser(data: data)
        parser.shouldProcessNamespaces = true; parser.shouldResolveExternalEntities = false; parser.delegate = reader
        guard parser.parse(), !reader.oversized else { throw AttachmentReader.failure("XML danneggiato o testo oltre 256 KiB.") }
        return reader.result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    func add(_ text: String, parser: XMLParser) {
        bytes += text.utf8.count
        if bytes > AttachmentReader.maxTextBytes { oversized = true; parser.abortParsing(); return }
        result += text
    }
    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes: [String : String] = [:]) {
        if kind == .word {
            guard ["http://schemas.openxmlformats.org/wordprocessingml/2006/main", "http://purl.oclc.org/ooxml/wordprocessingml/main"].contains(namespaceURI ?? "") else { return }
            if elementName == "t" { capture += 1 }
            if elementName == "tab" { add("\t", parser: parser) }
            if elementName == "br" || elementName == "cr" { add("\n", parser: parser) }
        } else {
            guard namespaceURI == "urn:oasis:names:tc:opendocument:xmlns:text:1.0" else { return }
            if elementName == "p" || elementName == "h" { capture += 1 }
            if elementName == "tab" { add("\t", parser: parser) }
            if elementName == "line-break" { add("\n", parser: parser) }
            if elementName == "s" {
                let count = Int(attributes.first(where: { $0.key == "c" || $0.key.hasSuffix(":c") })?.value ?? "1") ?? 0
                guard count > 0 && count <= AttachmentReader.maxTextBytes else { oversized = true; parser.abortParsing(); return }
                add(String(repeating: " ", count: count), parser: parser)
            }
        }
    }
    func parser(_ parser: XMLParser, foundCharacters string: String) { if capture > 0 { add(string, parser: parser) } }
    func parser(_ parser: XMLParser, foundCDATA CDATABlock: Data) { if capture > 0, let value = String(data: CDATABlock, encoding: .utf8) { add(value, parser: parser) } }
    func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
        if kind == .word {
            guard ["http://schemas.openxmlformats.org/wordprocessingml/2006/main", "http://purl.oclc.org/ooxml/wordprocessingml/main"].contains(namespaceURI ?? "") else { return }
            if elementName == "t" { capture = max(0, capture - 1) }
            if elementName == "p" { add("\n", parser: parser) }
            if elementName == "tc" { add("\t", parser: parser) }
        } else if namespaceURI == "urn:oasis:names:tc:opendocument:xmlns:text:1.0" && (elementName == "p" || elementName == "h") { capture = max(0, capture - 1); add("\n", parser: parser) }
    }
}
