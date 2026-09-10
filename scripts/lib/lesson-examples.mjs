import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {parse} from '@babel/parser';
import traverseModule from '@babel/traverse';
const traverse=traverseModule.default;

// Resolve actual JSX example references through their named data imports.
// Verification must not depend on authors naming every collection "ex".
export async function collectLessonExamples(filename){
 const absolute=path.resolve(filename),source=fs.readFileSync(absolute,'utf8');
 const ast=parse(source,{sourceType:'module',plugins:['jsx']});
 const imports=new Map(ast.program.body.filter(node=>node.type==='ImportDeclaration').flatMap(node=>node.specifiers.filter(specifier=>specifier.type==='ImportSpecifier').map(specifier=>[specifier.local.name,{source:node.source.value,export:specifier.imported.name}])));
 const references=[];
 traverse(ast,{JSXAttribute(attribute){
  if(attribute.node.name.name!=='example')return;
  const component=attribute.parentPath.node.name;
  if(component?.type!=='JSXIdentifier'||!['PythonExample','TerminalExample','RunnableExample'].includes(component.name))return;
  const expression=attribute.node.value?.expression;
  if(expression?.type!=='MemberExpression'||expression.object.type!=='Identifier')return;
  const key=expression.computed?expression.property.type==='StringLiteral'?expression.property.value:null:expression.property.name;
  const imported=imports.get(expression.object.name);if(!imported||!key)return;
  references.push({localName:expression.object.name,key,...imported});
 }});
 const resolved=[];
 for(const reference of references){
  const base=path.resolve(path.dirname(absolute),reference.source);const modulePath=[base,base+'.js'].find(candidate=>fs.existsSync(candidate)&&fs.statSync(candidate).isFile());
  if(!modulePath?.endsWith('.js'))throw new Error(`${filename}: expected a JavaScript data module for ${reference.localName}`);
  const namespace=await import(pathToFileURL(modulePath));const example=namespace[reference.export]?.[reference.key];
  if(!example||typeof example.code!=='string')throw new Error(`${filename}: missing executable example ${reference.localName}.${reference.key}`);
  resolved.push({collection:reference.export,key:reference.key,localName:reference.localName,example});
 }
 return resolved;
}
