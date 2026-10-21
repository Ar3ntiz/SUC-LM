const fs=require('fs'),path=require('path'),esbuild=require('esbuild');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'ui/index.html'),'utf8');
const catalog=JSON.parse(fs.readFileSync(path.join(root,'ui/localization.json'),'utf8'));
fs.writeFileSync(path.join(root,'app/src/main/assets/localization.json'),JSON.stringify(catalog));
const script=source.match(/<script>([\s\S]*?)<\/script>/)[1].replace('/*STUDIO_TRANSLATIONS*/{}',()=>JSON.stringify(catalog));
const result=esbuild.transformSync(script,{target:'es2017',charset:'utf8',minify:false});
fs.writeFileSync(path.join(root,'app/src/main/assets/index.html'),source.replace(/<script>[\s\S]*?<\/script>/,()=>'<script>'+result.code+'</script>'));
console.log('UI offline compiled for ES2017 WebView.');
