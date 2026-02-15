# phase 1: blueprint generation
async def Blueprint_Generation(Paper_R, config):
    # input
    # 自动识别输入类型（PDF URL Local），建立标准化deepcode_lab目录结构
    Analysis_Result = await run_research_analyzer(Paper_R)
    Dir_Info = await run_resource_processor(Analysis_Result)

    # Document Preprocessing, 根据文档长度决定是否进行语义分块
    # chunking -> (keyword, text)
    if should_use_document_segmentation(Paper_R):
        Chunks = await prepare_document_segments(Dir_info)
    else:
        Chunks = await read_full_document(Dir_info)
    
    # concept algorithm两个agent并行规划
    max_retries = 3
    for attempt in range(max_retries):
        # 并行调用两个agent
        [structure_plans, technical_specs] = await ParallelLLM.fan_out([
                Concept_Agent.analyze(Chunks),
                Algorithm_Agent.analyze(Chunks)
        ])
        
        # 聚合生成yaml格式的初步blueprint (codePlannerAgent)
        raw_blueprint = await Planning_Agent.fan_in(structure_plans, technical_specs)
        
        # 检查逻辑
        # 检查章节是否完整 yaml是否截断
        score = assess_output_completeness(raw_blueprint)
        
        if score >= 0.8:
                Blueprint_B = raw_blueprint
                break
        else:
        # 策略：减少 maxTokens 以为 input 留出空间，并降低 temperature 提高稳定性
                config.request_params = adjust_params_for_retry(config.request_params, attempt)
    # 检查是否仍有歧义 
    # hunman in the loop checkpoint，若有则通过人机交互解决歧义 
        if initial_structure.has_ambiguity():  
            questions = await Agent.generate_guiding_questions(initial_structure)
            # 向前端发送questionResponse
            user_answers = await Wait_For_User_Response(questions)
            Blueprint_B = await Agent.summarize(Paper_R, user_answers)

    return blueprint_B
		
# phase 2: Code generation

async def Agentic_Coding(Blueprint_B, External_Repo_J, config):

    # reference intelligence
    # 在enable_indexing开启时执行，搜索并下载论文引用的github库
    External_Repo_J = None
    if config.enable_indexing:
        ref_intel = await Reference_Intelligence_Agent.discover(Dir_Info)
        External_Repo_J = await Repository_Acquisition_Agent.download(ref_intel)
        await Codebase_Intelligence_Agent.index(External_Repo_J)

    # Sequential Synthesis with Memory
    Codebase_C = {}
    CodeMem_M = [] 
    
    for t, targetfile in enumerate(Blueprint_B.files):
    
        # 更新状态
        progress = 30 + int((t / len(Blueprint_B.files)) * 60)

    await Update_Task_Status(status="coding", progress=progress, message=f"Coding {target_file}")
            
        # SelectRelevantMemory
        # 识别当前目标文件相关的已实现模块，提取其结构化摘要
        relevant_context = SelectRelevantMemory(CodeMem_M, target_file)
        
        # CodeRAG, 外部检索
        rag_snippets = []
        if config.enable_indexing and Need_Reference(target_file):
                rag_snippets = await CodeRAG.retrieve(External_Repo_J, target_file)
                
        # code generation
        # 上下文 X_t = (Blueprint, 记忆摘要，RAG)
        X_t = Construct_Context(Blueprint_B, relevant_context, rag_snippets)
        code_t = await LLM.generate(X_t)
        
        # memory update
        summary_t = await Summary_Agent.extract(code_t)
        codeMem_M.append(summary_t)
        
        Codebase_C[target_file] = code_t
            
    return Codebase_C
    
# phase 3: Verification & Refinement
async def Automated_Refinement(Codebase_C, Blueprint_B):

    await Update_Task_Status(status="verifying", progress=90)
    # Static Analysis
    static_issues = await Analysis_Agent.check(Codebase_C, Blueprint_B)
    
    # sandbox execution
    success, trajectory =await Sandbox.run_repro_script(Codebase_C)
    
    # interative refinement
    attemps = 0
    while not success and attemps < MAX_RETRIES:
            # 分析轨迹并生成修改建议
            fix_suggestions = await Analysis_Agent.diagnose(trajectory, static_issues)
            
            # Modification agent进行行级修改
            Codebase_C = await Modification_Agent.apply_fix(Codebase_C, fix_suggestions)
            
            # 重新验证
            success, trajectory = await Sandbox.run_repro_script(Codebase_C)
            attempts += 1
				
    return Codebase_C