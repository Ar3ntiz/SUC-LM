import AppKit
import Foundation

enum ArtifactFiles {
    static let maximumSize=100*1024*1024
    static func url(_ artifact:ChatArtifact,archiveURL:URL) throws -> URL {
        guard !artifact.name.isEmpty,artifact.name != ".",artifact.name != "..",artifact.name == URL(fileURLWithPath:artifact.name).lastPathComponent,!artifact.name.contains("/"),!artifact.name.contains("\0") else{throw MCPConnection.failure("Nome artifact non valido.")}
        return archiveURL.deletingLastPathComponent().appendingPathComponent("artifacts",isDirectory:true).appendingPathComponent(artifact.id.uuidString,isDirectory:true).appendingPathComponent(artifact.name)
    }
    static func snapshot(path:String,root:URL,archiveURL:URL) throws -> ChatArtifact {
        let source=try WorkspaceTools.resolve(path,root:root)
        let values=try source.resourceValues(forKeys:[.isRegularFileKey,.fileSizeKey])
        guard values.isRegularFile==true,let size=values.fileSize,size<=maximumSize else{throw MCPConnection.failure("Artifact: serve un file regolare fino a 100 MiB.")}
        let artifact=ChatArtifact(name:source.lastPathComponent,byteCount:size)
        let destination=try url(artifact,archiveURL:archiveURL)
        try FileManager.default.createDirectory(at:destination.deletingLastPathComponent(),withIntermediateDirectories:true)
        do {
            try FileManager.default.copyItem(at:source,to:destination)
            guard (try destination.resourceValues(forKeys:[.fileSizeKey])).fileSize==size else{throw MCPConnection.failure("Il file è cambiato durante la pubblicazione.")}
        }catch{try? FileManager.default.removeItem(at:destination.deletingLastPathComponent());throw error}
        return artifact
    }
}
extension StudioStore {
    func publishArtifact(path:String,session:ChatSession) async throws -> String {
        let root=try workspaceURL(session),archiveURL=self.archiveURL
        let artifact=try await Task.detached {try ArtifactFiles.snapshot(path:path,root:root,archiveURL:archiveURL)}.value
        mutateSession(session.id){if $0.artifacts==nil{$0.artifacts=[]};$0.artifacts?.append(artifact)}
        scheduleSave()
        return "Artifact disponibile nella chat: \(artifact.name) (\(artifact.byteCount) byte)."
    }
    func artifactURL(_ artifact:ChatArtifact) throws -> URL {
        let url=try ArtifactFiles.url(artifact,archiveURL:archiveURL)
        guard FileManager.default.fileExists(atPath:url.path) else{throw MCPConnection.failure("Copia artifact non disponibile su questo Mac. Gli export JSON contengono solo i metadati: salva i file separatamente.")}
        return url
    }
    func saveArtifact(_ artifact:ChatArtifact){
        do{
            let source=try artifactURL(artifact),panel=NSSavePanel();panel.nameFieldStringValue=artifact.name
            guard panel.runModal() == .OK,let destination=panel.url else{return}
            guard destination.standardizedFileURL != source.standardizedFileURL else{return}
            // NSSavePanel asks before replacing an existing destination.
            if FileManager.default.fileExists(atPath:destination.path){let copy=try temporaryArtifactCopy(source);defer{try? FileManager.default.removeItem(at:copy)};_ = try FileManager.default.replaceItemAt(destination,withItemAt:copy)}
            else{try FileManager.default.copyItem(at:source,to:destination)}
        }catch{self.error=error.localizedDescription}
    }
    private func temporaryArtifactCopy(_ source:URL) throws -> URL {
        let copy=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at:source,to:copy);return copy
    }
    func openArtifact(_ artifact:ChatArtifact){do{let url=try artifactURL(artifact);NSWorkspace.shared.activateFileViewerSelecting([url])}catch{self.error=error.localizedDescription}}
}
