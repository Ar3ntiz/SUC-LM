import Foundation
import AppKit
@main struct ImageRevisionTests {
    @MainActor static func main() async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let bitmap=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:32,pixelsHigh:20,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0)!
        let png=bitmap.representation(using:.png,properties:[:])!;let fileURL=root.appendingPathComponent("image.custom");try png.write(to:fileURL)
        let file=try AttachmentReader.read(fileURL);assert(file.isImage && file.imageMIME=="image/jpeg" && file.imageWidth==32 && file.imageHeight==20)
        try AttachmentReader.validate([file]);let decoded=try JSONDecoder().decode(ChatAttachment.self,from:JSONEncoder().encode(file));assert(file==decoded)
        var bad=file;bad.imageData=Data([1,2,3]);do{try AttachmentReader.validate([bad]);assert(false)}catch{}
        let message=ChatMessage(role:"user",text:"Descrivi",attachments:[file]);let parts=message.remoteContent as! [[String:Any]]
        assert(parts.count==2 && parts[0]["text"] as? String == message.modelContent)
        assert(((parts[1]["image_url"] as! [String:String])["url"] ?? "").hasPrefix("data:image/jpeg;base64,"))
        let store=StudioStore(storageURL:root.appendingPathComponent("sessions.json"));store.modelState="ready";store.loadedModel="fixture"
        let session=store.selected!.id
        store.mutateSelected{$0.draft="Describe";$0.draftAttachments=[file]};store.sendMessage();assert(store.requests.isEmpty && store.error?.contains("vision")==true);assert(store.selected!.draftAttachments==[file])
        store.error=nil;store.mutateSelected{$0.draftAttachments=nil;$0.draft="Original"};store.sendMessage()
        let oldID=store.requests.keys.first!,oldUser=store.selected!.messages[0].id,oldAssistant=store.selected!.messages[1].id
        store.receive(["event":"delta","id":oldID,"payload":["text":"partial"]])
        await store.reviseMessage(oldAssistant,regenerate:true)
        assert(store.archive.sessions.count==1 && store.selected!.id==session && store.selected!.messages.count==2)
        let newID=store.requests.keys.first!;assert(newID != oldID)
        for event in ["delta","started","metrics","error","finished"] {store.receive(["event":event,"id":oldID,"payload":["text":"stale","message":"stale","completion_tokens":99]])}
        assert(store.error==nil && store.selected!.messages.last!.text.isEmpty && store.requests[newID] != nil)
        store.receive(["event":"delta","id":newID,"payload":["text":"new"]]);store.receive(["event":"finished","id":newID,"payload":[:]])
        assert(store.selected!.messages.last!.text=="new")
        let latestUser=store.selected!.messages[0].id;assert(latestUser != oldUser)
        await store.reviseMessage(latestUser,regenerate:false);assert(store.selected!.messages.isEmpty && store.selected!.draft=="Original" && store.selected!.id==session)
        assert(HookSamples.all.count==4 && HookSamples.all.allSatisfy{!$0.command.isEmpty})
        print("Image/revision tests passed: sniffing, JPEG, archive, multimodal payload, local rejection, same-session edit/reload and stale events")
    }
}
